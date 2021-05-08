//! Data structures and functions for adaptive octrees.

use super::{Octree, Statistics};
use ndarray::{Array1, ArrayView2, ArrayViewMut1, Axis};
use rusty_kernel_tools::RealType;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

fn refine_tree<T: RealType>(
    key: usize,
    refine_indices: &HashSet<usize>,
    mut particle_keys: ArrayViewMut1<usize>,
    particles: ArrayView2<T>,
    max_particles: usize,
    origin: &[f64; 3],
    diameter: &[f64; 3],
) {
    use crate::morton::{encode_point, find_level};

    let level = find_level(key);
    let mut new_keys = HashSet::<usize>::new();

    for &particle_index in refine_indices {
        let particle = [
            particles[[0, particle_index]].to_f64().unwrap(),
            particles[[1, particle_index]].to_f64().unwrap(),
            particles[[2, particle_index]].to_f64().unwrap(),
        ];

        let particle_key = encode_point(&particle, 1 + level, origin, diameter);
        particle_keys[particle_index] = particle_key;
        new_keys.insert(particle_key);
    }

    for new_key in new_keys {
        let associated_indices: HashSet<usize> = refine_indices
            .iter()
            .map(|&item| particle_keys[item])
            .filter(|&item| item == new_key)
            .collect();
        if associated_indices.len() > max_particles {
            refine_tree(
                new_key,
                &associated_indices,
                particle_keys.view_mut(),
                particles,
                max_particles,
                origin,
                diameter,
            );
        }
    }
}

/// Create a adaptive octree.
///
/// Returns a `AdaptiveOctree` struct describing an adaptive octree.
///
/// # Arguments
/// * `particles` - A (3, N) array of particles of type f32 or f64.
/// * `max_particles` - The maximum number of particles in each leaf.
pub fn adaptive_octree<T: RealType>(
    particles: ArrayView2<T>,
    max_particles: usize,
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

    adaptive_octree_with_bounding_box(particles, max_particles, origin, diameter)
}

/// Create an adaptive Octree with given bounding box.
///
/// Returns a `AdaptiveOctree` struct describing an adaptive octree.
///
/// # Arguments
/// * `particles` - A (3, N) array of particles of type f32 or f64.
/// * `max_particles` - Maximum number of particles.
/// * `origin` - The origin of the bounding box.
/// * `diameter` - The diameter of the bounding box in each dimension.
pub fn adaptive_octree_with_bounding_box<T: RealType>(
    particles: ArrayView2<T>,
    max_particles: usize,
    origin: [f64; 3],
    diameter: [f64; 3],
) -> Octree<'_, T> {
    use crate::morton::{compute_interaction_list, compute_near_field, find_level, find_parent};
    use rayon::prelude::*;
    use std::iter::FromIterator;
    let number_of_particles = particles.len_of(Axis(1));

    let now = Instant::now();

    let mut particle_keys = Array1::<usize>::zeros(number_of_particles);
    let refine_indices: HashSet<usize> = (0..number_of_particles).collect();

    refine_tree(
        0,
        &refine_indices,
        particle_keys.view_mut(),
        particles,
        max_particles,
        &origin,
        &diameter,
    );

    let max_level = particle_keys
        .iter()
        .map(|&item| find_level(item))
        .max()
        .unwrap();

    let nlevels = max_level + 1;

    let leaf_keys: HashSet<usize> = particle_keys.iter().copied().collect();

    let mut level_keys = HashMap::<usize, HashSet<usize>>::new();

    // Distribute the keys among different sets for each level
    for current_level in 0..nlevels {
        level_keys.insert(
            current_level,
            HashSet::from_iter(
                leaf_keys
                    .iter()
                    .cloned()
                    .filter(|&item| item == current_level),
            ),
        );
    }

    // Now fill up the sets with all the various parent keys.
    for current_level in (1..nlevels).rev() {
        let parent_index = current_level - 1;
        let current_set: HashSet<usize> = level_keys
            .get(&current_level)
            .unwrap()
            .iter()
            .copied()
            .collect();
        let parent_set = level_keys.get_mut(&parent_index).unwrap();
        parent_set.extend(current_set.iter().map(|&key| find_parent(key)));
    }

    let mut all_keys = HashSet::<usize>::new();

    for (_, current_keys) in &level_keys {
        all_keys.extend(current_keys.iter());
    }

    let mut leaf_key_to_particles = HashMap::<usize, HashSet<usize>>::new();
    for key in leaf_keys {
        leaf_key_to_particles.insert(key, HashSet::<usize>::new());
    }

    for (index, key) in particle_keys.iter().enumerate() {
        leaf_key_to_particles.get_mut(key).unwrap().insert(index);
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
        particle_keys: particle_keys,
        max_level: max_level,
        origin: origin,
        diameter: diameter,
        leaf_key_to_particles: leaf_key_to_particles,
        level_keys: level_keys,
        interaction_list: interaction_list,
        near_field: near_field,
        all_keys: all_keys,
        statistics: statistics,
    }
}
