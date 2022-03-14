use std::collections::{HashSet, HashMap};

use mpi::{
    Address,
    datatype::{
        Equivalence, UncommittedUserDatatype, UserDatatype
    },
    topology::{Color, Rank, UserCommunicator},
    environment::Universe,
    traits::*
};
use memoffset::offset_of;

use hyksort::hyksort::hyksort as hyksort;

use crate::{
    constants::ROOT,
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

    local_blocktree.keys.sort();
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

    let seeds: Vec<MortonKey> = complete
        .into_iter().filter(|k| k.level() == coarsest_level).collect();

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
            let ancestors = leaf.ancestors();
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
    &ncrit: &usize
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
            if npoints > ncrit {
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
    &ncrit: &usize,
    universe: &Universe,
    points: Vec<[PointType; 3]>,
    domain: &Domain,
    k: Rank,
) -> HashMap<MortonKey, MortonKey>{

    let comm = universe.world();
    let mut comm = comm.split_by_color(Color::with_value(0)).unwrap();
    let rank = comm.rank();
    let size = comm.size();

    // 1. Encode Points to Leaf Morton Keys
    // Map between points and keys
    let mut points: Vec<Point> = points
        .iter()
        .map(|p| Point{coordinate: p.clone(), global_idx: 0, key: MortonKey::from_point(&p, &domain)})
        .collect();

    // 2.i Perform parallel Morton sort over encoded points
    hyksort(&mut points, k, &mut comm);

    // // 2.ii, find unique leaves on each processor
    let leaves: Vec<MortonKey> = points
        .iter()
        .map(|p| p.key)
        .collect();

    let mut leaves = Tree {keys: leaves};
    leaves.linearize();
    leaves.sort();

    let comm = universe.world();
    let comm = comm.split_by_color(Color::with_value(0)).unwrap();

    // 3. Linearise received keys (remove overlaps if they exist).
    leaves.linearize();

    // 4. Complete region spanned by node.
    let mut tree = leaves;
    tree.complete();
    tree.sort();

    // 5. Find seeds and compute the coarse blocktree
    let mut seeds = find_seeds(&tree.keys);
    seeds.sort();

    let points = transfer_leaves_to_coarse_blocktree(
        &comm,
        &points,
        &seeds,
        &rank,
        &size
    );

    let mut blocktree = complete_blocktree(
        &mut seeds,
        &rank,
        &size,
        &comm
    );

    blocktree.sort();

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
    split_blocks(&leaves.keys, blocktree.keys, &ncrit)
}