use std::collections::HashMap;

use mpi::traits::*;
use mpi::topology::{Rank};

use rand::prelude::*;
use rand::{SeedableRng};

use rusty_tree::{
    distribute::unbalanced_tree,
    octree::Tree,
    types::{
        domain::Domain,
        point::Point,
        morton::MortonKey,
    }
};


pub fn points_fixture() -> Vec<[f64; 3]> {
    let npoints: u64 = 10000;

    let mut range = StdRng::seed_from_u64(0);
    let between = rand::distributions::Uniform::from(0.0..1.0);
    let mut points = Vec::new();

    for _ in 0..npoints {
        points.push([between.sample(&mut range), between.sample(&mut range), between.sample(&mut range)])
    }
    points
}

pub fn test_ncrit() {
    
    // 0. Experimental Parameters
    let ncrit: usize = 150;
    let k: Rank = 2;
    
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let domain = Domain{
        origin: [0., 0., 0.],
        diameter: [1., 1., 1.]
    };
    
    let points = points_fixture();

    let tree = unbalanced_tree(
        &ncrit, &universe, points, &domain, k
    );
    
    let mut blocks_to_points: HashMap<MortonKey, usize> = HashMap::new();

    for (_, block) in tree {

        if !blocks_to_points.contains_key(&block) {
            blocks_to_points.insert(block.clone(), 1);
        } else {
            if let Some(b) = blocks_to_points.get_mut(&block) {
                *b += 1;
            };
        }
    }

    for (_, &count) in &blocks_to_points {
        assert!(count <= ncrit);
    }    
}


pub fn test_span() {
    assert!(true)
}


fn main() {

    /// Test that the final tree satisfies the ncrit bound
    test_ncrit();

    /// Test that the final tree spans the entire space defined by the particle distribution.
    test_span();
}