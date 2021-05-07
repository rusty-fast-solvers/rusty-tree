//! Data structures and functions for regular octrees.
//! 
//! This module implements a regular Octree data structure.
//! A regular tree subdivides uniformly on all levels. To avoid
//! dealing with many empty boxes only the non-empty leaf boxes
//! are actually being stored.

use super::helpers::TreeStatistics;
use ndarray::{Array1, ArrayView2, Axis};
use rayon::prelude::*;
use rusty_kernel_tools::RealType;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

pub struct RegularOctree<'a, T: RealType> {
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
    pub particle_to_keys: Array1<usize>,

    /// The set of near-field keys for each non-empty key.
    pub near_field: HashMap<usize, HashSet<usize>>,

    /// The set of keys in the interaction list for each non-empty key.
    pub interaction_list: HashMap<usize, HashSet<usize>>,

    /// The index set of particles associated with each leaf key.
    pub leaf_key_to_particles: HashMap<usize, HashSet<usize>>,

    /// The set of all non-empty keys in the tree.
    pub all_keys: HashSet<usize>,

    /// Statistics for the tree.
    pub statistics: TreeStatistics,
}

/// Create a regular Octree.
///
/// Returns a `RegularOctree` struct describing a regular octree.
///
/// # Arguments
/// * `particles` - A (3, N) array of particles of type f32 or f64.
/// * `max_level` - The maximum level of the tree.
pub fn regular_octree<T: RealType>(
    particles: ArrayView2<T>,
    max_level: usize,
) -> RegularOctree<'_, T> {
    use super::helpers::compute_bounds;

    const TOL: f64 = 1E-5;

    let bounds = compute_bounds(particles);
    let diameter = [
        (bounds[0][1] - bounds[0][0]).to_f64().unwrap() * (1.0 + TOL),
        (bounds[1][1] - bounds[1][0]).to_f64().unwrap() * (1.0 + TOL),
        (bounds[2][1] - bounds[2][0]).to_f64().unwrap() * (1.0 + TOL),
    ];

    let origin = [
        bounds[0][0].to_f64().unwrap(),
        bounds[1][0].to_f64().unwrap(),
        bounds[2][0].to_f64().unwrap(),
    ];

    regular_octree_with_bounding_box(particles, max_level, origin, diameter)
}

/// Create a regular Octree with given bounding box.
///
/// Returns a `RegularOctree` struct describing a regular octree.
///
/// # Arguments
/// * `particles` - A (3, N) array of particles of type f32 or f64.
/// * `max_level` - The maximum level of the tree.
/// `origin` - The origin of the bounding box.
/// `diameter` - The diameter of the bounding box in each dimension.
pub fn regular_octree_with_bounding_box<T: RealType>(
    particles: ArrayView2<T>,
    max_level: usize,
    origin: [f64; 3],
    diameter: [f64; 3],
) -> RegularOctree<'_, T> {
    use super::morton::{compute_interaction_list, compute_near_field, encode_points, find_parent};
    use std::iter::FromIterator;

    let now = Instant::now();

    let nlevels: usize = 1 + max_level;

    let leaf_keys = encode_points(particles, max_level, &origin, &diameter);
    let mut leaf_key_to_particles = HashMap::<usize, HashSet<usize>>::new();

    for (particle_index, key) in leaf_keys.iter().enumerate() {
        if let Some(leaf_set) = leaf_key_to_particles.get_mut(key) {
            leaf_set.insert(particle_index);
        } else {
            let mut new_set = HashSet::<usize>::new();
            new_set.insert(particle_index);
            leaf_key_to_particles.insert(*key, new_set);
        }
    }

    let mut level_keys = HashMap::<usize, HashSet<usize>>::new();
    level_keys.insert(max_level, HashSet::from_iter(leaf_keys.iter().cloned()));

    for current_level in (1..nlevels).rev() {
        let mut parent_level = HashSet::<usize>::new();
        for &key in level_keys.get(&current_level).unwrap() {
            parent_level.insert(find_parent(key));
        }
        level_keys.insert(current_level - 1, parent_level);
    }

    let mut all_keys = HashSet::<usize>::new();

    for (_, current_keys) in &level_keys {
        all_keys = all_keys.union(&current_keys).cloned().collect();
    }

    let mut near_field = HashMap::<usize, HashSet<usize>>::new();
    let mut interaction_list = HashMap::<usize, HashSet<usize>>::new();

    for &key in &all_keys {
        near_field.insert(key, HashSet::<usize>::new());
        interaction_list.insert(key, HashSet::<usize>::new());
    }

    near_field.par_iter_mut().for_each(|(&key, hash_set)| {
        let current_near_field = compute_near_field(key);
        hash_set.extend(&current_near_field);
    });

    interaction_list
        .par_iter_mut()
        .for_each(|(&key, hash_set)| {
            let current_interaction_list = compute_interaction_list(key);
            hash_set.extend(&current_interaction_list);
        });

    let duration = now.elapsed();

    let statistics = TreeStatistics {
        number_of_particles: particles.len_of(Axis(1)),
        max_level: max_level,
        number_of_leafs: leaf_key_to_particles.keys().len(),
        number_of_keys: all_keys.len(),
        creation_time: duration,
        minimum_number_of_particles_in_leaf: leaf_key_to_particles
            .values()
            .map(|item| item.len())
            .reduce(std::cmp::min)
            .unwrap(),
        maximum_number_of_particles_in_leaf: leaf_key_to_particles
            .values()
            .map(|item| item.len())
            .reduce(std::cmp::max)
            .unwrap(),
        average_number_of_particles_in_leaf: (leaf_key_to_particles
            .values()
            .map(|item| item.len())
            .sum::<usize>() as f64)
            / (leaf_key_to_particles.keys().len() as f64),
    };

    RegularOctree {
        particles: particles,
        max_level: max_level,
        origin: origin,
        diameter: diameter,
        level_keys: level_keys,
        particle_to_keys: leaf_keys,
        near_field: near_field,
        interaction_list: interaction_list,
        leaf_key_to_particles: leaf_key_to_particles,
        all_keys: all_keys,
        statistics: statistics,
    }
}
