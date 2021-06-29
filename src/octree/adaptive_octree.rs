//! Data structures and functions for adaptive octrees.

use super::{Octree, OctreeType, Statistics};
use ndarray::{Array1, ArrayView2, ArrayViewMut1, Axis};
use rusty_kernel_tools::RealType;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

pub enum BalanceMode {
    /// Use for unbalanced adaptive octree.
    Unbalanced,
    /// Use for balanced adaptive octree.
    Balanced,
}

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

    if (level == 16) | (refine_indices.len() < max_particles) {
        // Do not refine if we have reached level cap or
        // we are already below the particle limit.
        return;
    }
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
            .copied()
            .filter(|&item| particle_keys[item] == new_key)
            .collect();
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

/// Create a adaptive octree.
///
/// Returns a `AdaptiveOctree` struct describing an adaptive octree.
///
/// # Arguments
/// * `particles` - A (3, N) array of particles of type f32 or f64.
/// * `max_particles` - The maximum number of particles in each leaf.
/// * `balance_mode` - Use `Balanced` for a 2:1 balanced octree, `Unbalanced` otherwise.
pub fn adaptive_octree<T: RealType>(
    particles: ArrayView2<T>,
    max_particles: usize,
    balance_mode: BalanceMode,
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

    adaptive_octree_with_bounding_box(particles, max_particles, origin, diameter, balance_mode)
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
/// * `balance_mode` - Use `Balanced` for a 2:1 balanced octree, `Unbalanced` otherwise.
pub fn adaptive_octree_with_bounding_box<T: RealType>(
    particles: ArrayView2<T>,
    max_particles: usize,
    origin: [f64; 3],
    diameter: [f64; 3],
    balance_mode: BalanceMode,
) -> Octree<'_, T> {
    use super::{
        compute_interaction_list_map, compute_leaf_map, compute_level_information,
        compute_near_field_map,
    };

    let number_of_particles = particles.len_of(Axis(1));

    let now = Instant::now();

    // First build up the non-adaptive tree by continuous refinement.

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

    let (max_level, mut all_keys, mut level_keys) = compute_level_information(particle_keys.view());

    match &balance_mode {
        BalanceMode::Balanced => balance_tree(
            &mut level_keys,
            particle_keys.view_mut(),
            particles,
            &mut all_keys,
            &origin,
            &diameter,
        ),
        _ => (),
    }

    let leaf_key_to_particles = compute_leaf_map(particle_keys.view());

    let near_field = compute_near_field_map(&all_keys);
    let interaction_list = compute_interaction_list_map(&all_keys);

    let duration = now.elapsed();

    let statistics = Statistics {
        number_of_particles: particles.len_of(Axis(1)),
        max_level,
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
        particles,
        particle_keys,
        max_level,
        origin,
        diameter,
        leaf_key_to_particles,
        level_keys,
        interaction_list,
        near_field,
        all_keys,
        octree_type: match &balance_mode {
            BalanceMode::Balanced => OctreeType::BalancedAdaptive,
            BalanceMode::Unbalanced => OctreeType::UnbalancedAdaptive,
        },
        statistics,
    }
}

/// Take a key and add the key and all its ancestors to the tree
fn find_completion(
    mut key: usize,
    level_keys: &mut HashMap<usize, HashSet<usize>>,
    all_keys: &mut HashSet<usize>,
) {
    use crate::morton::{find_level, find_parent};

    let mut intermediate_keys = HashSet::<usize>::new();
    let mut level = find_level(key);
    while !all_keys.contains(&key) {
        intermediate_keys.insert(key);
        level_keys.get_mut(&level).unwrap().insert(key);
        level = level - 1;
        key = find_parent(key);
    }

    all_keys.extend(intermediate_keys);
}

fn balance_tree<T: RealType>(
    level_keys: &mut HashMap<usize, HashSet<usize>>,
    mut particle_keys: ArrayViewMut1<usize>,
    particles: ArrayView2<T>,
    all_keys: &mut HashSet<usize>,
    origin: &[f64; 3],
    diameter: &[f64; 3],
) {
    use super::compute_complete_regular_tree;
    use crate::morton::{compute_near_field, encode_point, find_level, find_parent};

    let max_level = level_keys.keys().max().unwrap().clone();
    let nlevels = 1 + max_level;

    let regular_tree = compute_complete_regular_tree(particles, max_level, origin, diameter);

    for level in (1..nlevels).rev() {
        let current_keys: HashSet<usize> =
            level_keys.get(&level).unwrap().iter().copied().collect();
        for key in current_keys {
            let near_field = compute_near_field(key);
            for near_field_key in near_field {
                let parent = find_parent(near_field_key);
                // Only fill up if there can actually be particles in the parent
                // of the neighbour.
                if regular_tree.contains(&parent) {
                    find_completion(parent, level_keys, all_keys);
                }
            }
        }
    }

    // Now adapt the particle keys.
    for (particle_index, key) in particle_keys.iter_mut().enumerate() {
        let particle = [
            particles[[0, particle_index]].to_f64().unwrap(),
            particles[[1, particle_index]].to_f64().unwrap(),
            particles[[2, particle_index]].to_f64().unwrap(),
        ];

        let mut current_level = find_level(*key);

        while current_level < max_level {
            let descendent_key = encode_point(&particle, current_level + 1, origin, diameter);

            if all_keys.contains(&descendent_key) {
                *key = descendent_key;
                current_level += 1;
            } else {
                break;
            }
        }
    }
}
