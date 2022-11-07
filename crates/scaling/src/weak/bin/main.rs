use std::time::Instant;
use std::collections::{HashMap};

use mpi::{topology::SystemCommunicator, traits::*};
use mpi::collective::{SystemOperation};


use rand::prelude::*;
use rand::SeedableRng;

use rusty_tree::{
    distributed::DistributedTree,
};

const NPOINTS: u64 = 100000;

pub type Times = HashMap<String, u128>;


/// Test fixture for NPOINTS randomly distributed points.
fn points_fixture() -> Vec<[f64; 3]> {
    let mut range = StdRng::seed_from_u64(0);
    let between = rand::distributions::Uniform::from(0.0..1.0);
    let mut points = Vec::new();

    for _ in 0..NPOINTS {
        points.push([
            between.sample(&mut range),
            between.sample(&mut range),
            between.sample(&mut range),
        ])
    }
    points
}

/// Test fixture for an balanced tree.
fn balanced_tree_fixture(world: &SystemCommunicator) -> (DistributedTree, Times) {
    let points = points_fixture();
    let comm = world.duplicate();

    let mut times: Times = HashMap::new();
    let start  = Instant::now();
    let tree = DistributedTree::new(&points, true, &comm);
    times.insert("total".to_string(), start.elapsed().as_millis());
    (tree, times)
}


fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    // Distributed Trees
    let (balanced, times) = balanced_tree_fixture(&world);

    let root_rank = 0;
    let nleaves = balanced.keys.len();
    let mut sum = 0;

    // Print runtime to stdout
    if rank == root_rank {
        world
            .process_at_rank(root_rank)
            .reduce_into_root(&nleaves, &mut sum, SystemOperation::sum());
        
        // universe size, number of leaves, total runtime, encoding time, sorting time
        println!(
            "{:?}, {:?}, {:?}",
            size,
            sum,
            times.get(&"total".to_string()),
        )
    } else {
        world
            .process_at_rank(root_rank)
            .reduce_into(&nleaves, SystemOperation::sum())
    }

}