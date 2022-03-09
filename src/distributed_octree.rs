use std::collections::{HashSet, HashMap};

use mpi::{
    Address,
    Count,
    datatype::{
        Equivalence, UncommittedUserDatatype, UserDatatype
    },
    topology::{Color, Process, Rank, UserCommunicator},
    environment::Universe,
    traits::*
};
use memoffset::offset_of;

use hyksort::hyksort::hyksort as hyksort;

use crate::ROOT;
use crate::morton::{MortonKey, Point};
use crate::types::{KeyType, PointType, Domain};
use crate::serial_octree::{Tree, LinearTree, CompleteLinearTree};


#[derive(Debug, Clone)]
pub struct BlockBounds {
    pub rank: Rank,
    pub lower: MortonKey,
    pub upper: MortonKey,
}

unsafe impl Equivalence for BlockBounds {
    type Out = UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[1, 1, 1],
            &[
                offset_of!(BlockBounds, rank) as Address,
                offset_of!(BlockBounds, lower) as Address,
                offset_of!(BlockBounds, upper) as Address
            ],
            &[
                UncommittedUserDatatype::contiguous(1, &Rank::equivalent_datatype()).as_ref(),
                UncommittedUserDatatype::contiguous(4, &MortonKey::equivalent_datatype()).as_ref(),
                UncommittedUserDatatype::contiguous(4, &MortonKey::equivalent_datatype()).as_ref(),
            ]
        )
    }
}


/// Complete a distributed blocktree from the seed octants, algorithm 4 in [1] (parallel).
pub fn complete_blocktree(
    seeds: &mut Vec<MortonKey>,
    &rank: &Rank,
    &size: &Rank,
    world: &UserCommunicator,
) -> CompleteLinearTree {
    if rank == 0 {
        let dfd_root = ROOT.finest_first_child();
        let min = seeds.iter().min().unwrap();
        let na = dfd_root.finest_ancestor(&min);
        let first_child = na.children().into_iter().min().unwrap();
        seeds.push(first_child);
    }

    if rank == (size - 1) {
        let dld_root = ROOT.finest_last_child();
        let max = seeds.iter().max().unwrap();
        let na = dld_root.finest_ancestor(&max);
        let last_child = na.children().into_iter().max().unwrap();
        seeds.push(last_child);
    }

    seeds.sort();

    let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
    let previous_rank = if rank > 0 { rank - 1 } else { size - 1 };

    let previous_process = world.process_at_rank(previous_rank);
    let next_process = world.process_at_rank(next_rank);

    // Send required data to partner process.
    if rank > 0 {
        let min = seeds.iter().min().unwrap().clone();
        previous_process.send(&min);
    }

    if rank < (size - 1) {
        let mut rec = MortonKey::default();
        next_process.receive_into(&mut rec);
        seeds.push(rec);
    }

    // Complete region between seeds at each process
    let mut local_blocktree = CompleteLinearTree{keys: Vec::new()};

    for i in 0..(seeds.iter().len() - 1) {
        let a = seeds[i];
        let b = seeds[i + 1];

        let mut tmp: Vec<MortonKey> = LinearTree::complete_region(&a, &b);

        // if rank == 3 {
        //     let max_seed = seeds.iter().max().unwrap();
        //     assert!(!tmp.contains(&max_seed))
        //     // println!(
        //         //     "i {:?} a {:?} b {:?} tmp {:?} max seed {:?}",
        //         //     i, a, b, tmp, seeds.iter().max().unwrap()
        //         // );
        // }

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
        min_seed = points.iter().min().unwrap().morton;
    } else {
        min_seed = seeds.iter().min().unwrap().clone();
    }

    let prev_rank = if rank > 0 { rank - 1 } else {size-1};
    let next_rank = if rank +1 < size { rank + 1 } else { 0 };

    if rank > 0 {
        let msg: Vec<Point> = points
            .iter()
            .filter(|&p| p.morton < min_seed)
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

    // Filter out stuff that's been sent to partner
    let mut points: Vec<Point> = points
        .iter()
        .filter(|&p| p.morton >= min_seed)
        .cloned()
        .collect();


    received_points.append(&mut points);

    received_points.sort();
    received_points
}

pub fn assign_blocks_to_points(
    leaves: &Vec<MortonKey>,
    blocktree: &Vec<MortonKey>,
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
    let mut points: Vec<Point> = points
        .iter()
        .map(|p| Point{coordinate: p.clone(), morton: MortonKey::from_point(&p, &domain)})
        .collect();


    // 2.i Perform parallel Morton sort over encoded points
    hyksort(&mut points, k, &mut comm);
    // println!("Rank {:?}, points {:?}", rank, points.iter().len());

    // // 2.ii, find unique leaves on each processor
    let mut leaves: Vec<MortonKey> = points
        .iter()
        .map(|p| p.morton)
        .collect();

    let mut leaves = Tree::from_iterable(
        leaves.into_iter()
    ).linearize();

    leaves.keys.sort();

    println!(
        "Rank {:?}, complete {:?} min {:?}",
        rank, points.iter().min().unwrap().morton, points.iter().max().unwrap().morton
    );

    // let comm = universe.world();
    // let comm = comm.split_by_color(Color::with_value(0)).unwrap();

    // // 3. Linearise received keys (remove overlaps if they exist).
    // leaves =  leaves.linearize();

    // // 4. Complete region spanned by node.
    // let tree = leaves.complete();
    // // println!("Rank {:?}, complete {:?}", rank, tree.keys.iter().len());

    // let tree = Tree::from_iterable(tree.keys.clone().into_iter());

    // // // 5. Find seeds and compute the coarse blocktree
    // let seed_level = tree.keys.iter().map(|k| k.level()).min().unwrap();
    // let mut seeds: Vec<MortonKey> = tree.keys.iter().filter(|k| k.level() == seed_level).cloned().collect();
    // // println!("Rank {:?}, seed level {:?}", rank, seed_level);

    // let seed_domain = [seeds.iter().min().unwrap(), seeds.iter().max().unwrap()];

    // // assert!(seed_domain[0] <= &points.iter().min().unwrap().morton);
    // // assert!(seed_domain[1] >= &points.iter().max().unwrap().morton);
    // // println!("RANK {:?}, seeds {:?}", rank, seed_level);

    // let mut blocktree = complete_blocktree(
    //     &mut seeds,
    //     &rank,
    //     &size,
    //     &comm
    // );

    // blocktree.keys.sort();

    // let block_domain = [blocktree.keys.iter().min().unwrap(), blocktree.keys.iter().max().unwrap()];
    // println!("RANK {:?}, block_domain {:?}", rank, blocktree);

//     // 5.ii any data below the min seed sent to partner process

//     let points = transfer_leaves_to_coarse_blocktree(
//         &comm,
//         &points,
//         &seeds,
//         &rank,
//         &size
//     );
//     let leaves: Vec<MortonKey> = points
//         .iter()
//         .map(|p| p.morton)
//         .collect();

//     let leaves = Tree::from_iterable(
//         leaves.into_iter()
//     ).linearize();

// //     // // 6. Match particle data to blocks

//     // let point_domain = [leaves.keys.iter().min().unwrap(), leaves.keys.iter().max().unwrap()];
//     let point_domain = [points.iter().min().unwrap().morton,points.iter().max().unwrap().morton];
//     let block_domain = [blocktree.keys.iter().min().unwrap().clone(), blocktree.keys.iter().max().unwrap().clone()];

//     if rank == 0 {
//         // assert!(point_domain[0] >= block_domain[0]);
//         if !(point_domain[1] <= block_domain[1]) {
//             println!("point domain {:?} \n block domain {:?} greater {:?} ",
//             point_domain[1], block_domain[1], point_domain[1] <= block_domain[1]
//         );
//     }
// }
//     let points_to_blocks = assign_blocks_to_points(
//         &leaves.keys,
//         &blocktree.keys
//     );

//     // println!("rank {:?} points_to_blocks {:?}", rank, leaves.keys.iter().len())

//     for point in points.iter() {
//         if points_to_blocks.get(&point.morton).is_none() {
//             println!("fuck")
//         };
//     }

//     println!("rank {:?} points to blocks {:?}", rank, points_to_blocks.get(&points[0].morton).unwrap().level());

    // 7. Refine blocks based on ncrit


}