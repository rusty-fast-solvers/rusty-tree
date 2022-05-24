use rusty_tree::data::HDF5;
use rusty_tree::distributed::DistributedTree;
use rusty_tree::types::morton::MortonKey;
use rusty_tree::types::point::Point;

use mpi::{topology::SystemCommunicator, traits::*};

use rand::prelude::*;
use rand::SeedableRng;
const NPOINTS: u64 = 200;

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

/// Test fixture for an unbalanced tree.
fn unbalanced_tree_fixture(world: &SystemCommunicator) -> DistributedTree {
    let points = points_fixture();

    let comm = world.duplicate();

    DistributedTree::new(&points, false, &comm)
}

/// Test fixture for an balanced tree.
fn balanced_tree_fixture(world: &SystemCommunicator) -> DistributedTree {
    let points = points_fixture();
    let comm = world.duplicate();

    DistributedTree::new(&points, true, &comm)
}
fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let balanced = balanced_tree_fixture(&world);

    let comm = world.duplicate();
    // DistributedTree::write_hdf5(&comm, "foo".to_string(), &balanced);

    let tree = DistributedTree::read_hdf5(&comm, "foo.hdf5".to_string());

    println!("rank {:?} tree {:?}", world.rank(), tree.keys.len())
}
