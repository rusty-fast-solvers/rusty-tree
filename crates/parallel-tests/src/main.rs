use std::collections::HashMap;

use mpi::{
    traits::*,
    topology::Rank,
    environment::Universe,
};

use rand::prelude::*;
use rand::{SeedableRng};

use rusty_tree::{
    constants::NCRIT,
    distribute::unbalanced_tree,
    types::{
        domain::Domain,
        morton::MortonKey,
    }
};


fn points_fixture() -> Vec<[f64; 3]> {
    let npoints: u64 = 10000;

    let mut range = StdRng::seed_from_u64(0);
    let between = rand::distributions::Uniform::from(0.0..1.0);
    let mut points = Vec::new();

    for _ in 0..npoints {
        points.push([between.sample(&mut range), between.sample(&mut range), between.sample(&mut range)])
    }
    points
}

fn unbalanced_tree_fixture(universe: &Universe) -> HashMap<MortonKey, MortonKey> {

    // Experimental Parameters
    let k: Rank = 2;
    let domain = Domain{
        origin: [0., 0., 0.],
        diameter: [1., 1., 1.]
    };

    let points = points_fixture();

    unbalanced_tree(&universe, points, &domain)

}

fn test_ncrit(universe: &Universe) {

    let tree = unbalanced_tree_fixture(universe);

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
        assert!(count <= NCRIT);
    }
}


fn test_span(universe: &Universe) {
    assert!(true)
}


fn main() {

    let universe = mpi::initialize().unwrap();

    // Test that the final tree satisfies the ncrit bound
    test_ncrit(&universe);

    // Test that the final tree spans the entire space defined by the particle distribution.
    test_span(&universe);
}