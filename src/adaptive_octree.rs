//! Data structures and functions for adaptive octrees.

use ndarray::{ArrayView2, Axis};
use rayon::prelude::*;
use rusty_kernel_tools::RealType;
use std::collections::{HashMap, HashSet};

pub struct AdaptiveOctree<'a, T: RealType> {
    /// A (3, N) array of N particles.
    pub particles: ArrayView2<'a, T>,

    /// The maximum level in the tree.
    pub max_level: usize,

    /// The origin of the bounding box for the particles.
    pub origin: [f64; 3],

    /// The diameter across each dimension of the bounding box.
    pub diameter: [f64; 3],

    /// Mapping from keys to associated particle indices.
    pub keys_to_indices: HashMap<usize, HashSet<usize>>,
}

fn refine_partition<T: RealType>(
    key: usize,
    particle_indices: &HashSet<usize>,
    particles: ArrayView2<T>,
    key_to_indices: &mut HashMap<usize, HashSet<usize>>,
    max_particles: usize,
    origin: &[f64; 3],
    diameter: &[f64; 3],
) {
    use super::morton::{encode_point, find_level};

    let level = find_level(key);

    key_to_indices.insert(key, HashSet::<usize>::new());
    key_to_indices
        .get_mut(&key)
        .unwrap()
        .extend(particle_indices);

    if particle_indices.len() < max_particles {
        return;
    }

    let mut local_map = HashMap::<usize, HashSet<usize>>::new();

    for &particle_index in particle_indices {
        let particle = [
            particles[[0, particle_index]].to_f64().unwrap(),
            particles[[1, particle_index]].to_f64().unwrap(),
            particles[[2, particle_index]].to_f64().unwrap(),
        ];

        let particle_key = encode_point(&particle, 1 + level, origin, diameter);

        if !local_map.contains_key(&particle_key) {
            local_map.insert(particle_key, HashSet::<usize>::new());
        }

        local_map
            .get_mut(&particle_key)
            .unwrap()
            .insert(particle_index);
    }

    let children_maps: Vec<HashMap<usize, HashSet<usize>>> = local_map
        .par_iter()
        .map(
            |(my_key, indices): (&usize, &HashSet<usize>)| -> HashMap<usize, HashSet<usize>> {
                let mut new_map = HashMap::<usize, HashSet<usize>>::new();
                refine_partition(
                    *my_key,
                    indices,
                    particles,
                    &mut new_map,
                    max_particles,
                    origin,
                    diameter,
                );
                new_map
            },
        )
        .collect();
    for child_map in children_maps.into_iter() {
        for (child_key, child_set) in child_map.into_iter() {
            key_to_indices.insert(child_key, child_set);
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
) -> AdaptiveOctree<'_, T> {
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
) -> AdaptiveOctree<'_, T> {
    use super::morton::find_level;
    let number_of_particles = particles.len_of(Axis(1));

    let mut keys_to_indices = HashMap::<usize, HashSet<usize>>::new();
    let particle_indices: HashSet<usize> = (0..number_of_particles).collect();

    refine_partition(
        0,
        &particle_indices,
        particles,
        &mut keys_to_indices,
        max_particles,
        &origin,
        &diameter,
    );

    let max_level = keys_to_indices
        .keys()
        .map(|&item| find_level(item))
        .max()
        .unwrap();

    AdaptiveOctree {
        particles: particles,
        max_level: max_level,
        origin: origin,
        diameter: diameter,
        keys_to_indices: keys_to_indices,
    }

    // let statistics = TreeStatistics {
    //     number_of_particles: particles.len_of(Axis(1)),
    //     max_level: max_level,
    //     number_of_leafs: leaf_key_to_particles.keys().len(),
    //     number_of_keys: all_keys.len(),
    //     creation_time: duration,
    //     minimum_number_of_particles_in_leaf: leaf_key_to_particles
    //         .values()
    //         .map(|item| item.len())
    //         .reduce(std::cmp::min)
    //         .unwrap(),
    //     maximum_number_of_particles_in_leaf: leaf_key_to_particles
    //         .values()
    //         .map(|item| item.len())
    //         .reduce(std::cmp::max)
    //         .unwrap(),
    //     average_number_of_particles_in_leaf: (leaf_key_to_particles
    //         .values()
    //         .map(|item| item.len())
    //         .sum::<usize>() as f64)
    //         / (leaf_key_to_particles.keys().len() as f64),
    // };
}
