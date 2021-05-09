//! Data structures and functions for regular octrees.
//! 
//! This module implements a regular Octree data structure.
//! A regular tree subdivides uniformly on all levels. To avoid
//! dealing with many empty boxes only the non-empty leaf boxes
//! are actually being stored.

use super::{Statistics, Octree, OctreeType};
use ndarray::{ArrayView2, Axis};
use rusty_kernel_tools::RealType;
use std::time::Instant;


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
) -> Octree<'_, T> {
    use crate::helpers::compute_bounds;

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
) -> Octree<'_, T> {
    use crate::morton::{encode_points};
    use super::{compute_near_field_map, compute_interaction_list_map, compute_leaf_map, compute_level_information};

    let now = Instant::now();


    let leaf_keys = encode_points(particles, max_level, &origin, &diameter);
    let (max_level, all_keys, level_keys) = compute_level_information(leaf_keys.view());
    let leaf_key_to_particles = compute_leaf_map(leaf_keys.view());

    let near_field = compute_near_field_map(&all_keys);
    let interaction_list = compute_interaction_list_map(&all_keys);

    let duration = now.elapsed();

    let statistics = Statistics {
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

    Octree {
        particles: particles,
        max_level: max_level,
        origin: origin,
        diameter: diameter,
        level_keys: level_keys,
        particle_keys: leaf_keys,
        near_field: near_field,
        interaction_list: interaction_list,
        leaf_key_to_particles: leaf_key_to_particles,
        all_keys: all_keys,
        octree_type: OctreeType::Regular,
        statistics: statistics,
    }
}
