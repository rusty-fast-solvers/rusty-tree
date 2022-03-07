use mpi::{
    topology::{Color, Process, Rank, UserCommunicator},
    environment::Universe,
    point_to_point as p2p,
    traits::*
};

use hyksort::hyksort::hyksort as hyksort;

use crate::ROOT;
use crate::morton::{MortonKey};
use crate::types::{KeyType, PointType, Domain};
use crate::serial_octree::{Tree, LinearTree, CompleteLinearTree};


/// Complete a distributed blocktree from the seed octants, algorithm 4 in [1] (parallel).
pub fn complete_blocktree(
    seeds: &mut Vec<MortonKey>,
    &rank: &Rank,
    &size: &Rank,
    world: UserCommunicator,
) -> CompleteLinearTree {
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
        let min = *seeds.iter().min().unwrap();
        previous_process.send(&min);
    }

    if rank < (size - 1) {
        let mut rec = MortonKey::default();
        next_process.receive_into(&mut rec);
        seeds.push(rec);
    }

    // Complete region between seeds at each process
    let mut local_blocktree = CompleteLinearTree{keys: Vec::new()};

    for i in 0..(seeds.len() - 1) {
        let a = seeds[i];
        let b = seeds[i + 1];

        let mut tmp = LinearTree::complete_region(&a, &b);
        local_blocktree.keys.push(a);
        local_blocktree.keys.append(&mut tmp);
    }

    if rank == (size - 1) {
        local_blocktree.keys.push(*seeds.last().unwrap());
    }

    local_blocktree.keys.sort();
    local_blocktree
}

pub fn unbalanced_tree(
    depth: &KeyType,
    ncrit: &KeyType,
    &size: &Rank,
    &rank: &Rank,
    universe: &Universe,
    points: Vec<[PointType; 3]>,
    domain: &Domain,
    k: Rank,
) {

    let comm = universe.world();
    let mut comm = comm.split_by_color(Color::with_value(0)).unwrap();

    // 1. Encode Points to Leaf Morton Keys

    // Map between points and keys
    let keys_to_points: Vec<MortonKey> = points
        .iter()
        .map(|p| MortonKey::from_point(&p, &domain))
        .collect();

    let tree = Tree::from_iterable(keys_to_points.into_iter());
    let mut tree = tree.linearize();

    // 2. Perform parallel Morton sort over leaves
    hyksort(&mut tree.keys, k, &mut comm);
    let comm = universe.world();
    let comm = comm.split_by_color(Color::with_value(0)).unwrap();

    // 3. Linearise received keys (remove overlaps if they exist).
    let tree =  tree.linearize();

    // 4. Complete region spanned by node.
    let tree = tree.complete();

    // 5. Find seeds
    let seed_level = tree.keys.iter().map(|k| k.level()).min().unwrap();
    let mut seeds: Vec<MortonKey> = tree.keys.iter().filter(|k| k.level() == seed_level).cloned().collect();
    // seeds.sort();
    // println!("rank {:?} seeds {:?}", rank, seeds);
    // 6. Compute region between seeds at each process

    if rank == 0 {
        println!("domain {:?} {:?}", seeds.iter().min(), seeds.iter().max());
    }
    if rank == 1 {
        println!("min {:?}", seeds.iter().min());
    }

    let blocktree = complete_blocktree(
        &mut seeds,
        &rank,
        &size,
        comm
    );

    // println!("rank {:?} blocktree {:?}", rank, blocktree.keys.iter().len());

    // println!("rank {:?} blocks {:?}", rank, blocktree.keys)

}