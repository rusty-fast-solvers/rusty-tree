use rand::Rng;
use mpi::traits::*;
use mpi::topology::{Color, Rank};

use rusty_tree::distributed_octree::{unbalanced_tree};
use rusty_tree::morton::{MortonKey};
use rusty_tree::types::Domain;


fn main() {

    // Setup MPI
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let mut world = world.split_by_color(Color::with_value(0)).unwrap();
    let size = world.size();
    let rank = world.rank();
    let root_rank = 0;

    // 0. Experimental Parameters
    let depth: u64 = 5;
    let ncrit: u64 = 150;
    let npoints: u64 = 100;
    let k: Rank = 2;

    let domain = Domain{
        origin: [0., 0., 0.],
        diameter: [1., 1., 1.]
    };

    let mut range = rand::thread_rng();
    let mut points = Vec::new();

    for _ in 0..npoints {
        points.push([range.gen(), range.gen(), range.gen()]);
    }

    // let keys: Vec<MortonKey> = points
    //     .iter()
    //     .map(|p| MortonKey::from_point(&p, &domain))
    //     .collect();

    // println!("keys {:?}", keys);

    let tree = unbalanced_tree(
        &depth, &ncrit, &size, &rank, &universe, points, &domain, k
    );

}