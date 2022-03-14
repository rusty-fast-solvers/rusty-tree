use rand::prelude::*;
use rand::{SeedableRng};
use mpi::topology::{Rank};

use rusty_tree::distribute::unbalanced_tree;
use rusty_tree::types::domain::Domain;


fn main() {

    // Setup MPI
    let universe = mpi::initialize().unwrap();

    // 0. Experimental Parameters
    let ncrit: usize = 150;
    let npoints: u64 = 10000;
    let k: Rank = 2;

    let domain = Domain{
        origin: [0., 0., 0.],
        diameter: [1., 1., 1.]
    };

    let mut range = StdRng::seed_from_u64(0);
    let between = rand::distributions::Uniform::from(0.0..1.0);
    let mut points = Vec::new();

    for _ in 0..npoints {
        points.push([between.sample(&mut range), between.sample(&mut range), between.sample(&mut range)])
    }

    let tree = unbalanced_tree(
        &ncrit, &universe, points, &domain, k
    );

    // Test ncrit
    {
        use mpi::traits::*;
        use rusty_tree::types::morton::MortonKey;
        use std::collections::HashMap;

        let world = universe.world();
        let rank = world.rank();

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

        if rank == 0 {
            println!("blocks_{:?}=np.array([", rank);
            for (block,_) in &blocks_to_points {
                println!("    [{:?}, {:?}, {:?}, {:?}],",block.anchor[0], block.anchor[1], block.anchor[2], block.level());
            }
            println!("]) \n");
        }

    }
}