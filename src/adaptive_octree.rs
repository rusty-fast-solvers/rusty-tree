//! Data structures and functions for adaptive octrees.

use super::helpers::TreeStatistics;
use ndarray::{Array1, ArrayView2, Axis};
use rayon::prelude::*;
use rusty_kernel_tools::RealType;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

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
    max_elements: usize,
    origin: &[f64; 3],
    diameter: &[f64; 3],
) {
    use super::morton::{encode_point, find_level};

    let level = find_level(key);

    key_to_indices.insert(key, HashSet::<usize>::new());
    key_to_indices.get_mut(&key).unwrap().extend(particle_indices);

    if particle_indices.len() < max_elements {
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

    let mut new_maps = Vec::<HashMap<usize, HashSet<usize>>>::new();
    for _ in 0..local_map.len() {
        new_maps.push(HashMap::<usize, HashSet<usize>>::new());
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
                    max_elements,
                    origin,
                    diameter,
                );
                new_map
            },
        )
        .collect();
    
    for child_map in children_maps {
        key_to_indices.extend(child_map);

    }
}
