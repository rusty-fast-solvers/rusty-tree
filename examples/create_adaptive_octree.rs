//! This example demonstrates the creation of a regular octree.

use ndarray::{Array2, Axis};
use rand;
use rand::Rng;
use rusty_tree::octree::*;

const NPARTICLES: usize = 10000;
const MAX_PARTICLES: usize = 100;

pub fn main() {
    // Create random particles on the unit sphere.

    let mut rng = rand::thread_rng();
    let mut particles = Array2::<f64>::zeros((3, NPARTICLES));
    particles.map_inplace(|item| *item = rng.gen::<f64>());

    for mut particle in particles.axis_iter_mut(Axis(1)) {
        let norm = particle.iter().map(|item| item.powi(2)).sum::<f64>().sqrt();
        particle.map_inplace(|item| *item = *item / norm);
    }

    let balanced_tree = adaptive_octree(particles.view(), MAX_PARTICLES, BalanceMode::Balanced);

    export_to_vtk(&balanced_tree, "./test.vtk");

    println!("{}", balanced_tree.statistics);
}
