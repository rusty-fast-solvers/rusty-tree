//! Data structures and functions to create regular and adaptive Octrees.

use ndarray::{Array1, ArrayView2};
use rusty_kernel_tools::RealType;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::Duration;

pub mod adaptive_octree;
pub mod regular_octree;

pub use adaptive_octree::*;
pub use regular_octree::*;

/// The type of the octree.
pub enum OctreeType {
    /// Regular octree.
    Regular,
    /// Use for balanced adaptive octrees.
    BalancedAdaptive,
    /// Use for unbalanced adaptive octrees.
    UnbalancedAdaptive,
}

/// The basic Octree data structure
pub struct Octree<'a, T: RealType> {
    /// A (3, N) array of N particles.
    pub particles: ArrayView2<'a, T>,

    /// The maximum level in the tree.
    pub max_level: usize,

    /// The origin of the bounding box for the particles.
    pub origin: [f64; 3],

    /// The diameter across each dimension of the bounding box.
    pub diameter: [f64; 3],

    /// The non-empty keys for each level of the tree.
    pub level_keys: HashMap<usize, HashSet<usize>>,

    /// The keys associated with the particles.
    pub particle_keys: Array1<usize>,

    /// The set of near-field keys for each non-empty key.
    pub near_field: HashMap<usize, HashSet<usize>>,

    /// The set of keys in the interaction list for each non-empty key.
    pub interaction_list: HashMap<usize, HashSet<usize>>,

    /// The index set of particles associated with each leaf key.
    pub leaf_key_to_particles: HashMap<usize, HashSet<usize>>,

    /// The set of all non-empty keys in the tree.
    pub all_keys: HashSet<usize>,

    /// The type of the Octree.
    pub octree_type: OctreeType,

    /// Statistics for the tree.
    pub statistics: Statistics,
}

/// A structure that stores various statistics for a tree.
pub struct Statistics {
    pub number_of_particles: usize,
    pub max_level: usize,
    pub number_of_leafs: usize,
    pub number_of_keys: usize,
    pub creation_time: Duration,
    pub minimum_number_of_particles_in_leaf: usize,
    pub maximum_number_of_particles_in_leaf: usize,
    pub average_number_of_particles_in_leaf: f64,
}

impl std::fmt::Display for Statistics {
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
