use std::collections::{HashMap, HashSet};

use mpi::{environment::Universe, topology::Rank, traits::*};

use rand::prelude::*;
use rand::SeedableRng;

use rusty_tree::{
    constants::{NCRIT, ROOT},
    distribute::{balanced_tree, unbalanced_tree},
    types::{domain::Domain, morton::MortonKey},
};

const NPOINTS: u64 = 10000;

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

fn unbalanced_tree_fixture(universe: &Universe) -> HashMap<MortonKey, MortonKey> {
    // Experimental Parameters
    let k: Rank = 2;
    let domain = Domain {
        origin: [0., 0., 0.],
        diameter: [1., 1., 1.],
    };

    let points = points_fixture();

    unbalanced_tree(&universe, points, &domain)
}

fn balanced_tree_fixture(universe: &Universe) -> HashMap<MortonKey, MortonKey> {
    // Experimental Parameters
    let k: Rank = 2;
    let domain = Domain {
        origin: [0., 0., 0.],
        diameter: [1., 1., 1.],
    };

    let points = points_fixture();

    balanced_tree(&universe, points, &domain)
}

/// Test that the tree satisfies the ncrit condition.
fn test_ncrit(tree: &HashMap<MortonKey, MortonKey>) {
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

/// Test that the tree spans the entire domain specified by the point distribution.
fn test_span(tree: &HashMap<MortonKey, MortonKey>, rank: Rank) {
    let min = tree.iter().map(|(_, block)| block).min().unwrap();
    let max = tree.iter().map(|(_, block)| block).max().unwrap();
    let block_set: HashSet<MortonKey> = tree.iter().map(|(_, block)| block.clone()).collect();
    let max_level = tree.iter().map(|(_, block)| block.level()).max().unwrap();

    // Generate a uniform tree at the max level, and filter for range in this processor
    let mut level = 0;
    let mut uniform = vec![ROOT.clone()];
    while level < max_level {
        let mut descendents: Vec<MortonKey> = Vec::new();

        for node in uniform.iter() {
            let mut children = node.children();
            descendents.append(&mut children);
        }

        uniform = descendents;

        level += 1;
    }

    uniform = uniform
        .into_iter()
        .filter(|node| min <= node && node <= max)
        .collect();

    // Test that each member of the uniform tree, or their ancestors are contained within the
    // tree.
    for node in uniform.iter() {
        let ancestors = node.ancestors();

        let int: Vec<MortonKey> = ancestors
            .intersection(&block_set)
            .into_iter()
            .cloned()
            .collect();

        assert!(int.iter().len() > 0);
    }
}


fn test_no_overlaps(universe: &Universe, tree: &HashMap<MortonKey, MortonKey>) {

    let world = universe.world();

    // Communicate bounds from each process
    let max = tree.iter().map(|(_, block)| block).max().unwrap();
    let min = tree.iter().map(|(_, block)| block).min().unwrap().clone();

    // Gather all bounds at root
    let size = world.size();
    let rank = world.rank();

    let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
    let previous_rank = if rank > 0 { rank - 1 } else { size - 1 };

    let previous_process = world.process_at_rank(previous_rank);
    let next_process = world.process_at_rank(next_rank);

    // Send min to partner
    if rank > 0 {
        previous_process.send(&min);
    }

    let mut partner_min = MortonKey::default();

    if rank < (size - 1) {
        next_process.receive_into(&mut partner_min);
    }

    // Test that the partner's minimum node is greater than the process's maximum node
    if rank < size -1 {
        assert!(max <= &partner_min)
    }
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    // Distributed Trees
    let unbalanced = unbalanced_tree_fixture(&universe);
    let balanced = balanced_tree_fixture(&universe);

    // Tests for the unbalanced tree
    {
        test_ncrit(&unbalanced);
        test_span(&unbalanced, rank);
        test_no_overlaps(&universe, &unbalanced);
    }

    // Tests for the balanced tree
    {
        test_ncrit(&balanced);
        test_span(&balanced, rank);
        test_no_overlaps(&universe, &balanced);
    }

}
