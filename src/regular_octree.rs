//! This module implements a regular Octree data structure.
//! A regular tree subdivides uniformly on all levels. To avoid
//! dealing with many empty boxes only the non-empty leaf boxes
//! are actually being stored.

use ndarray::{Array1, ArrayView2};
use rusty_kernel_tools::RealType;
use std::collections::{HashMap, HashSet};

pub struct RegularOctree<'a, T: RealType> {
    pub particles: ArrayView2<'a, T>,
    pub max_level: usize,
    pub origin: [f64; 3],
    pub diameter: [f64; 3],
    pub level_keys: HashMap<usize, HashSet<usize>>,
    pub particle_to_keys: Array1<usize>,
    pub near_field: HashMap<usize, HashSet<usize>>,
    pub interaction_list: HashMap<usize, HashSet<usize>>,
    pub all_keys: HashSet<usize>,
}

pub fn regular_octree<T: RealType>(particles: ArrayView2<T>, max_level: usize) -> RegularOctree<'_, T>{
    use super::helpers::compute_bounds;
    use super::morton::{encode_points, find_parent, compute_near_field, compute_interaction_list};
    use std::iter::FromIterator;

    const TOL: f64 = 1E-5;

    let bounds = compute_bounds(particles);
    let nlevels = 1 + max_level;


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

    let leaf_keys = encode_points(particles, max_level, &origin, &diameter);

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
        near_field.insert(key, compute_near_field(key));
        interaction_list.insert(key, compute_interaction_list(key));
    }

    RegularOctree {
        particles: particles,
        max_level: max_level,
        origin: origin,
        diameter: diameter,
        level_keys: level_keys,
        particle_to_keys: leaf_keys,
        near_field: near_field,
        interaction_list: interaction_list,
        all_keys: all_keys,
    }


}
