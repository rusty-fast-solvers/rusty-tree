//! This example demonstrates the creation of a regular octree.

use ndarray::{Array2, Axis};
use rand;
use rand::Rng;
use rusty_tree::regular_octree::regular_octree;

const NPARTICLES: usize = 1000000;
const MAX_LEVEL: usize = 6;

pub fn main() {
    // Create random particles on the unit sphere.

    let mut rng = rand::thread_rng();
    let mut particles = Array2::<f64>::zeros((3, NPARTICLES));
    particles.map_inplace(|item| *item = rng.gen::<f64>());

    for mut particle in particles.axis_iter_mut(Axis(1)) {
        let norm = particle.iter().map(|item| item.powi(2)).sum::<f64>().sqrt();
        particle.map_inplace(|item| *item = *item / norm);
    }

    let tree = regular_octree(particles.view(), MAX_LEVEL);

    println!("{}", tree.statistics);
}
