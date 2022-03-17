//! Data structures and functions to create distributed Octrees with MPI.
use std::collections::{HashSet, HashMap};

use mpi::{
    topology::{Color, Rank, UserCommunicator},
    environment::Universe,
    traits::*
};

use hyksort::hyksort::hyksort as hyksort;

use crate::{
    constants::{ROOT, NCRIT, K},
    single_node::Tree,
    types::{
        domain::Domain,
        morton::MortonKey,
        point::{Point, PointType},
    }
};

pub struct DistributedTree {
    pub balanced: bool,
    pub points: Vec<Point>,
    pub keys_to_nodes: HashMap<MortonKey, MortonKey>,
    pub keys: Vec<MortonKey>
}

impl DistributedTree {

    pub fn new(points: &[[PointType; 3]], domain: &Domain, balanced: bool, universe: &Universe) -> DistributedTree {

        if balanced {
            let (points, keys_to_nodes) = DistributedTree::balanced_tree(universe, points, domain);
            let keys = points.iter().map(|p| p.key).collect();

            DistributedTree {
                balanced,
                points,
                keys,
                keys_to_nodes,
            }
        } else {
            let (points, keys_to_nodes) = DistributedTree::unbalanced_tree(universe, points, domain);
            let keys = points.iter().map(|p| p.key).collect();

            DistributedTree {
                balanced,
                points,
                keys,
                keys_to_nodes,
            }
        }
    }

    /// Complete a distributed block tree from the seed octants, algorithm 4 in [1] (parallel).
    fn complete_blocktree(
        seeds: &mut Vec<MortonKey>,
        &rank: &Rank,
        &size: &Rank,
        world: &UserCommunicator,
    ) -> Tree {

        // Define the tree's global domain with the finest first/last descendants
        if rank == 0 {
            let ffc_root = ROOT.finest_first_child();
            let min = seeds.iter().min().unwrap();
            let fa = ffc_root.finest_ancestor(min);
            let first_child = fa.children().into_iter().min().unwrap();
            seeds.push(first_child);
            seeds.sort();
        }

        if rank == (size - 1) {
            let flc_root = ROOT.finest_last_child();
            let max = seeds.iter().max().unwrap();
            let fa = flc_root.finest_ancestor(max);
            let last_child = fa.children().into_iter().max().unwrap();
            seeds.push(last_child);
        }

        let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
        let previous_rank = if rank > 0 { rank - 1 } else { size - 1 };

        let previous_process = world.process_at_rank(previous_rank);
        let next_process = world.process_at_rank(next_rank);

        // Send required data to partner process.
        if rank > 0 {
            let min = *seeds.iter().min().unwrap();
            previous_process.send(&min);
        }

        let mut boundary = MortonKey::default();

        if rank < (size - 1) {
            next_process.receive_into(&mut boundary);
            seeds.push(boundary);
        }

        // Complete region between seeds at each process
        let mut complete = Tree {keys: Vec::new()};

        for i in 0..(seeds.iter().len() - 1) {
            let a = seeds[i];
            let b = seeds[i + 1];

            let mut tmp: Vec<MortonKey> = Tree::complete_region(&a, &b);
            complete.keys.push(a);
            complete.keys.append(&mut tmp);
        }

        if rank == (size - 1) {
            complete.keys.push(*seeds.last().unwrap());
        }

        complete.sort();
        complete
    }

    fn assign_nodes_to_leaves(
        leaves: &Vec<MortonKey>,
        nodes: Vec<MortonKey>,
    ) -> HashMap<MortonKey, MortonKey> {

        let nodes: HashSet<MortonKey> = nodes.iter().cloned().collect();
        let mut map : HashMap<MortonKey, MortonKey> = HashMap::new();

        for leaf in leaves.iter() {

            if nodes.contains(leaf) {
                map.insert(*leaf, *leaf);
            } else {
                let mut ancestors: Vec<MortonKey> = leaf.ancestors().into_iter().collect();
                ancestors.sort();
                for ancestor in ancestors {
                    if nodes.contains(&ancestor) {
                        map.insert(*leaf, ancestor);
                        break;
                    }
                }
            };
        }
        map
    }

    fn split_blocks(
        leaves: &Vec<MortonKey>,
        mut blocktree: Vec<MortonKey>,
    ) -> HashMap<MortonKey, MortonKey> {

        let mut split_blocktree: Vec<MortonKey> = Vec::new();

        loop {
            let mut new_blocktree: Vec<MortonKey> = Vec::new();

            // Map between blocks and the leaves they contain
            let blocks_to_leaves = DistributedTree::assign_nodes_to_leaves(
                leaves,
                blocktree.clone(),
            );

            // Count the number of points in a block
            let mut blocks_to_npoints: HashMap<MortonKey, usize> = HashMap::new();
            for (_, block) in blocks_to_leaves {
                if let std::collections::hash_map::Entry::Vacant(e) = blocks_to_npoints.entry(block) {
                    e.insert(1);
                } else if let Some(b) = blocks_to_npoints.get_mut(&block) {
                        *b += 1;       
                }
            }

            // Generate a new blocktree with a block's children if they violate the NCRIT constraint
            let mut check = 0;
            for (&block, &npoints) in blocks_to_npoints.iter() {
                if npoints > NCRIT {
                    let mut children = block.children();
                    new_blocktree.append(&mut children);
                } else {
                    new_blocktree.push(block);
                    check += 1;
                }
            }

            if check == blocks_to_npoints.len() {
                split_blocktree = new_blocktree;
                break;
            } else {
                blocktree = new_blocktree;
            }
        }

        // Assign nodes in the split blocktree to all leaves
        DistributedTree::assign_nodes_to_leaves(
            leaves,
            split_blocktree,
        )
    }

    /// Find the seeds at each processor [1].
    fn find_seeds(leaves: &[MortonKey]) -> Vec<MortonKey> {

        let min: MortonKey = *leaves.iter().min().unwrap();
        let max: MortonKey = *leaves.iter().max().unwrap();

        // Complete the region between the least and greatest leaves.
        let mut complete = Tree::complete_region(&min, &max);
        complete.push(min);
        complete.push(max);

        // Find seeds by filtering for leaves at coarsest level
        let coarsest_level = complete.iter().map(|k| k.level()).min().unwrap();
        let mut seeds: Vec<MortonKey> = complete
            .into_iter()
            .filter(|k| k.level() == coarsest_level)
            .collect();

        seeds.sort();
        seeds
    }

    // Transfer points based on the coarse distributed blocktree [REFERENCE ALGORITHM]
    fn transfer_points_to_blocktree(
        comm: &UserCommunicator,
        points: &[Point],
        seeds: &[MortonKey],
        &rank: &Rank,
        &size:& Rank,
    ) -> Vec<Point> {

        let mut received_points : Vec<Point> = Vec::new();

        let mut min_seed = MortonKey::default();

        if rank == 0 {
            min_seed = points.iter().min().unwrap().key;
        } else {
            min_seed = *seeds.iter().min().unwrap();
        }

        let prev_rank = if rank > 0 { rank - 1 } else {size-1};
        let next_rank = if rank +1 < size { rank + 1 } else { 0 };

        if rank > 0 {
            let msg: Vec<Point> = points
                .iter()
                .filter(|&p| p.key < min_seed)
                .cloned()
                .collect();

            let msg_size: Rank = msg.len() as Rank;
            comm.process_at_rank(prev_rank).send(&msg_size);
            comm.process_at_rank(prev_rank).send(&msg[..]);
        }

        if rank < (size - 1) {
            let mut bufsize = 0;
            comm.process_at_rank(next_rank).receive_into(&mut bufsize);
            let mut buffer = vec![Point::default(); bufsize as usize];
            comm.process_at_rank(next_rank).receive_into(&mut buffer[..]);
            received_points.append(&mut buffer);
        }

        // Filter out local points that's been sent to partner
        let mut points: Vec<Point> = points
            .iter()
            .filter(|&p| p.key >= min_seed)
            .cloned()
            .collect();

        received_points.append(&mut points);
        received_points.sort();

        received_points
    }

    pub fn unbalanced_tree(
        universe: &Universe,
        points: &[[PointType; 3]],
        domain: &Domain,
    ) -> (Vec<Point>, HashMap<MortonKey, MortonKey>) {

        let comm = universe.world();
        let mut comm = comm.split_by_color(Color::with_value(0)).unwrap();
        let rank = comm.rank();
        let size = comm.size();

        // 1. Encode Points to Leaf Morton Keys, add a global index related to the processor
        let mut points: Vec<Point> = points
            .iter()
            .enumerate()
            .map(|(i, p)| Point{coordinate: *p, global_idx: i, key: MortonKey::from_point(p, domain)})
            .collect();

        // 2.i Perform parallel Morton sort over encoded points
        hyksort(&mut points, K, &mut comm);

        // 2.ii Find unique leaf keys on each processor and place in a Tree
        let keys: Vec<MortonKey> = points
            .iter()
            .map(|p| p.key)
            .collect();

        let mut tree = Tree {keys};

        // 3. Linearise received keys (remove overlaps if they exist).
        tree.linearize();

        let comm = universe.world();
        let comm = comm.split_by_color(Color::with_value(0)).unwrap();

        // 4. Complete region spanned by node.
        tree.complete();

        // 5. Find seeds and compute the coarse blocktree
        let mut seeds = DistributedTree::find_seeds(&tree.keys);

        let blocktree = DistributedTree::complete_blocktree(
            &mut seeds,
            &rank,
            &size,
            &comm
        );

        // 5.ii any data below the min seed sent to partner process
        let points = DistributedTree::transfer_points_to_blocktree(
            &comm,
            &points,
            &seeds,
            &rank,
            &size
        );

        // 6. Refine blocks based on ncrit
        let map = DistributedTree::split_blocks(
            &points.iter().map(|p| p.key).collect(),
            blocktree.keys
        );

        (points, map)
    }

    pub fn balanced_tree(
        universe: &Universe,
        points: &[[PointType; 3]],
        domain: &Domain,
    ) -> (Vec<Point>, HashMap<MortonKey, MortonKey>) {

        // Create a distributed unbalanced tree;
        let comm = universe.world();
        let mut comm = comm.split_by_color(Color::with_value(0)).unwrap();
        let rank = comm.rank();
        let size = comm.size();

        // 1. Encode Points to Leaf Morton Keys, add a global index related to the processor
        let mut points: Vec<Point> = points
            .iter()
            .enumerate()
            .map(|(i, p)| Point{coordinate: *p, global_idx: i, key: MortonKey::from_point(p, domain)})
            .collect();

        // 2.i Perform parallel Morton sort over encoded points
        hyksort(&mut points, K, &mut comm);

        // 2.ii Find unique leaf keys on each processor and place in a Tree
        let keys: Vec<MortonKey> = points
            .iter()
            .map(|p| p.key)
            .collect();

        let mut tree = Tree {keys};

        // 3. Linearise received keys (remove overlaps if they exist).
        tree.linearize();

        let comm = universe.world();
        let comm = comm.split_by_color(Color::with_value(0)).unwrap();

        // 4. Complete region spanned by node.
        tree.complete();

        // 5.i Find seeds and compute the coarse blocktree
        let mut seeds = DistributedTree::find_seeds(&tree.keys);

        let blocktree = DistributedTree::complete_blocktree(
            &mut seeds,
            &rank,
            &size,
            &comm
        );

        // 5.ii Send data below the min seed sent to partner process
        let points = DistributedTree::transfer_points_to_blocktree(
            &comm,
            &points,
            &seeds,
            &rank,
            &size
        );

        // 6. Refine blocks based on ncrit
        let unbalanced_tree = DistributedTree::split_blocks(
            &points.iter().map(|p| p.key).collect(),
            blocktree.keys
        );

        // 7.i Create the minimal balanced tree for local octants, spanning the entire domain, and linearize
        let linearized = Tree {
            keys: unbalanced_tree.into_iter().map(|(_, block)| block).collect()
        };
        linearized.balance();

        // 7.ii Assign the local leaves to the elements of this new balanced tree
        let points_to_balanced = DistributedTree::assign_nodes_to_leaves(
            &points.iter().map(|p| p.key).collect(),
            linearized.keys,
        );

        let mut points: Vec<Point> = points
            .iter()
            .map(|p| Point{
                coordinate: p.coordinate,
                global_idx: p.global_idx,
                key: *points_to_balanced.get(&p.key).unwrap()
            })
            .collect();

        // 8. Perform another distributed sort
        let comm = universe.world();
        let mut comm = comm.split_by_color(Color::with_value(0)).unwrap();
        hyksort(&mut points, K, &mut comm);

        // 9. Remove local overlaps
        let keys: Vec<MortonKey> = points
            .iter()
            .map(|p| p.key)
            .collect();

        let mut tree = Tree{keys};
        tree.linearize();

        // 10. Map points to non-overlapping tree
        let map = DistributedTree::assign_nodes_to_leaves(
            &points.iter().map(|p| p.key).collect(),
            tree.keys
        );

        (points, map)
    }
}
