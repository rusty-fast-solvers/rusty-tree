use rand::prelude::*;
use rand::{Rng, SeedableRng, distributions::Uniform};
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
    let ncrit: usize = 150;
    let npoints: u64 = 10000;
    let k: Rank = 2;

    let domain = Domain{
        origin: [0., 0., 0.],
        diameter: [1., 1., 1.]
    };

    // let mut range: Rng = SeedableRng::from_seed(0);
    let mut range = StdRng::seed_from_u64(0);
    let between = rand::distributions::Uniform::from(0.0..1.0);
    let mut points = Vec::new();

    for _ in 0..npoints {
        points.push([between.sample(&mut range), between.sample(&mut range), between.sample(&mut range)])
    }

    // let keys: Vec<MortonKey> = points
    //     .iter()
    //     .map(|p| MortonKey::from_point(&p, &domain))
    //     .collect();

    // println!("keys {:?}", keys);

    let tree = unbalanced_tree(
        &ncrit, &size, &rank, &universe, points, &domain, k
    );

}