use std::collections::{HashSet, HashMap};

use mpi::{
    topology::{Color, Rank, UserCommunicator},
    environment::Universe,
    traits::*
};

use hyksort::hyksort::hyksort as hyksort;

use crate::{
    constants::{ROOT, NCRIT, K, DEEPEST_LEVEL},
    octree::Tree,
    types::{
        domain::Domain,
        morton::MortonKey,
        point::{Point, PointType},
    }
};

/// Complete a distributed blocktree from the seed octants, algorithm 4 in [1] (parallel).
pub fn complete_blocktree(
    seeds: &mut Vec<MortonKey>,
    &rank: &Rank,
    &size: &Rank,
    world: &UserCommunicator,
) -> Tree {
    if rank == 0 {
        let dfd_root = ROOT.finest_first_child();
        let min = seeds.iter().min().unwrap();
        let na = dfd_root.finest_ancestor(&min);
        let first_child = na.children().into_iter().min().unwrap();
        seeds.push(first_child);
        seeds.sort();
    }

    if rank == (size - 1) {
        let dld_root = ROOT.finest_last_child();
        let max = seeds.iter().max().unwrap();
        let na = dld_root.finest_ancestor(&max);
        let last_child = na.children().into_iter().max().unwrap();
        seeds.push(last_child);
    }

    let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
    let previous_rank = if rank > 0 { rank - 1 } else { size - 1 };

    let previous_process = world.process_at_rank(previous_rank);
    let next_process = world.process_at_rank(next_rank);

    // Send required data to partner process.
    if rank > 0 {
        let min = seeds.iter().min().unwrap().clone();
        previous_process.send(&min);
    }

    let mut boundary = MortonKey::default();

    if rank < (size - 1) {
        next_process.receive_into(&mut boundary);
        seeds.push(boundary);
    }

    // Complete region between seeds at each process
    let mut local_blocktree = Tree {keys: Vec::new()};

    for i in 0..(seeds.iter().len() - 1) {
        let a = seeds[i];
        let b = seeds[i + 1];

        let mut tmp: Vec<MortonKey> = Tree::complete_region(&a, &b);
        local_blocktree.keys.push(a);
        local_blocktree.keys.append(&mut tmp);
    }

    if rank == (size - 1) {
        local_blocktree.keys.push(seeds.last().unwrap().clone());
    }

    local_blocktree.sort();
    local_blocktree
}

pub fn transfer_leaves_to_coarse_blocktree(
    comm: &UserCommunicator,
    points: &Vec<Point>,
    seeds: &Vec<MortonKey>,
    &rank: &Rank,
    &size:& Rank,
) -> Vec<Point> {

    let mut received_points : Vec<Point> = Vec::new();

    let mut min_seed = MortonKey::default();

    if rank == 0 {
        min_seed = points.iter().min().unwrap().key;
    } else {
        min_seed = seeds.iter().min().unwrap().clone();
    }

    let prev_rank = if rank > 0 { rank - 1 } else {size-1};
    let next_rank = if rank +1 < size { rank + 1 } else { 0 };

    if rank > 0 {
        let msg: Vec<Point> = points
            .iter()
            .filter(|&p| p.key < min_seed)
            .cloned()
            .collect();
        // println!("here msg size {:?}", msg.iter().len());
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

    // Filter out stuff that's been sent to partner
    let mut points: Vec<Point> = points
        .iter()
        .filter(|&p| p.key >= min_seed)
        .cloned()
        .collect();

    received_points.append(&mut points);

    received_points.sort();
    received_points
}

pub fn find_seeds(local_leaves: &Vec<MortonKey>) -> Vec<MortonKey> {

    let min: MortonKey = local_leaves.iter().min().unwrap().clone();
    let max: MortonKey = local_leaves.iter().max().unwrap().clone();

    // Complete the region between the least and greatest leaves.
    let mut complete = Tree::complete_region(&min, &max);
    complete.push(min);
    complete.push(max);

    // Find blocks
    let coarsest_level = complete.iter().map(|k| k.level()).min().unwrap();

    let mut seeds: Vec<MortonKey> = complete
        .into_iter().filter(|k| k.level() == coarsest_level).collect();

    seeds.sort();
    seeds
}

pub fn assign_blocks_to_points(
    leaves: &Vec<MortonKey>,
    blocktree: Vec<MortonKey>,
) -> HashMap<MortonKey, MortonKey> {

    let blocktree_set: HashSet<MortonKey> = blocktree.iter().cloned().collect();
    let mut map : HashMap<MortonKey, MortonKey> = HashMap::new();

    for leaf in leaves.iter() {

        if blocktree_set.contains(leaf) {
            map.insert(leaf.clone(), leaf.clone());
        } else {
            let mut ancestors: Vec<MortonKey> = leaf.ancestors().into_iter().collect();
            ancestors.sort();
            for ancestor in ancestors {
                if blocktree_set.contains(&ancestor) {
                    map.insert(leaf.clone(), ancestor);
                    break;
                }
            }
        };
    }
    map
}

pub fn split_blocks(
    leaves: &Vec<MortonKey>,
    mut blocktree: Vec<MortonKey>,
) -> HashMap<MortonKey, MortonKey> {

    let mut refined = false;
    let mut unbalanced_tree: Vec<MortonKey> = Vec::new();

    loop {
        let points_to_blocks = assign_blocks_to_points(
            leaves,
            blocktree.clone(),
        );

        let mut blocks_to_points: HashMap<MortonKey, usize> = HashMap::new();
        let mut new_blocktree: Vec<MortonKey> = Vec::new();

        for (_, block) in points_to_blocks {

            if !blocks_to_points.contains_key(&block) {
                blocks_to_points.insert(block.clone(), 1);
            } else {
                if let Some(b) = blocks_to_points.get_mut(&block) {
                    *b += 1;
                };
            }
        }

        let mut check = 0;
        for (&block, &npoints) in blocks_to_points.iter() {
            if npoints > NCRIT {
                let mut children = block.children();
                new_blocktree.append(&mut children);
            } else {
                new_blocktree.push(block);
                check += 1;
            }
        }

        if check == blocks_to_points.len() {
            unbalanced_tree = blocktree.to_vec();
            break;
        } else {
            blocktree = new_blocktree;
        }
    }

    let points_to_blocks = assign_blocks_to_points(
        leaves,
        unbalanced_tree,
    );

    points_to_blocks
}


pub fn unbalanced_tree(
    universe: &Universe,
    points: Vec<[PointType; 3]>,
    domain: &Domain,
) -> HashMap<MortonKey, MortonKey>{

    let comm = universe.world();
    let mut comm = comm.split_by_color(Color::with_value(0)).unwrap();
    let rank = comm.rank();
    let size = comm.size();

    // 1. Encode Points to Leaf Morton Keys, add a global index related to the processor
    let mut points: Vec<Point> = points
        .iter()
        .enumerate()
        .map(|(i, p)| Point{coordinate: p.clone(), global_idx: i, key: MortonKey::from_point(&p, &domain)})
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
    let mut seeds = find_seeds(&tree.keys);

    let blocktree = complete_blocktree(
        &mut seeds,
        &rank,
        &size,
        &comm
    );

    // 5.ii any data below the min seed sent to partner process
    let points = transfer_leaves_to_coarse_blocktree(
        &comm,
        &points,
        &seeds,
        &rank,
        &size
    );

    let keys: Vec<MortonKey> = points
        .iter()
        .map(|p| p.key)
        .collect();

    let mut leaves = Tree{keys};
    leaves.linearize();

    // 6. Refine blocks based on ncrit
    split_blocks(&leaves.keys, blocktree.keys)
}

pub fn balanced_tree(
    universe: &Universe,
    points: Vec<[PointType; 3]>,
    domain: &Domain,
) -> HashMap<MortonKey, MortonKey> {

    // Create a distributed unbalanced tree;
    let comm = universe.world();
    let mut comm = comm.split_by_color(Color::with_value(0)).unwrap();
    let rank = comm.rank();
    let size = comm.size();

    // 1. Encode Points to Leaf Morton Keys, add a global index related to the processor
    let mut points: Vec<Point> = points
        .iter()
        .enumerate()
        .map(|(i, p)| Point{coordinate: p.clone(), global_idx: i, key: MortonKey::from_point(&p, &domain)})
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
    let mut seeds = find_seeds(&tree.keys);

    let blocktree = complete_blocktree(
        &mut seeds,
        &rank,
        &size,
        &comm
    );

    // 5.ii any data below the min seed sent to partner process
    let points = transfer_leaves_to_coarse_blocktree(
        &comm,
        &points,
        &seeds,
        &rank,
        &size
    );

    let keys: Vec<MortonKey> = points
        .iter()
        .map(|p| p.key)
        .collect();

    let leaves = Tree{keys};

    // 6. Refine blocks based on ncrit
    let unbalanced_tree = split_blocks(&leaves.keys, blocktree.keys);

    // 7.i Create the minimal balanced tree for local octants, spanning the entire domain, and linearize
    let mut linearized = Tree {
        keys: unbalanced_tree.into_iter().map(|(_, block)| block).collect()
    };
    linearized.balance();

    // 7.ii Assign the local points to the elements of this new balanced tree
    let points_to_balanced = assign_blocks_to_points(
        &leaves.keys,
        linearized.keys,
    );

    let mut points: Vec<Point> = points
        .iter()
        .map(|p| Point{
            coordinate: p.coordinate,
            global_idx: p.global_idx,
            key: points_to_balanced.get(&p.key).unwrap().clone()
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
    assign_blocks_to_points(
        &points.iter().map(|p| p.key).collect(),
        tree.keys,
    )
}