//! This module contains useful helper functions.

use ndarray::{ArrayView2, Axis};
use num;
use rusty_kernel_tools::RealType;
use std::fmt;
use std::time::Duration;

/// Compute the bounds of a (3, N) array
///
/// This function returns a (3, 2) array `bounds`, where `bounds[i][0]`
/// is the lower bound along the ith axis, and `bounds[i][1]` is the
/// upper bound along the jth axis.
///
/// # Arguments
///
/// * `arr` - A (3, N) array.
///
pub fn compute_bounds<T: RealType>(arr: ArrayView2<T>) -> [[T; 2]; 3] {
    let mut bounds: [[T; 2]; 3] = [[num::traits::zero(); 2]; 3];

    arr.axis_iter(Axis(0)).enumerate().for_each(|(i, axis)| {
        bounds[i][0] = axis.iter().copied().reduce(T::min).unwrap();
        bounds[i][1] = axis.iter().copied().reduce(T::max).unwrap()
    });

    bounds
}

/// A structure that stores various statistics for a tree.
pub struct TreeStatistics {
    pub number_of_particles: usize,
    pub max_level: usize,
    pub number_of_leafs: usize,
    pub number_of_keys: usize,
    pub creation_time: Duration,
    pub minimum_number_of_particles_in_leaf: usize,
    pub maximum_number_of_particles_in_leaf: usize,
    pub average_number_of_particles_in_leaf: f64,
}

impl std::fmt::Display for TreeStatistics {
    /// Create an output string for tree statistics.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "\n\nOctree Statistics\n\
                   ==============================\n\
                   Number of Particles: {}\n\
                   Maximum level: {}\n\
                   Number of leaf keys: {}\n\
                   Number of keys in tree: {}\n\
                   Creation time [s]: {}\n\
                   Minimum number of particles in leaf node: {}\n\
                   Maximum number of particles in leaf node: {}\n\
                   Average number of particles in leaf node: {:.2}\n\
                   ==============================\n\n
                   ",
            self.number_of_particles,
            self.max_level,
            self.number_of_leafs,
            self.number_of_keys,
            (self.creation_time.as_millis() as f64) / 1000.0,
            self.minimum_number_of_particles_in_leaf,
            self.maximum_number_of_particles_in_leaf,
            self.average_number_of_particles_in_leaf
        )
    }
}
