// //! Data structures and functions to create regular and adaptive Octrees.

//! Algorithms for serial Octrees

use crate::{
    constants::DEEPEST_LEVEL,
    types::morton::Key as Key
};

use itertools::Itertools;
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
pub struct Tree {
    pub keys: HashSet<Key>,
}

#[derive(Debug)]
pub struct LinearTree {
    pub keys: Vec<Key>,
}

#[derive(Debug)]
pub struct CompleteLinearTree {
    pub keys: Vec<Key>,
}

impl Tree {
    pub fn from_iterable<T: Iterator<Item = Key>>(keys: T) -> Tree {

        let mut key_set = HashSet::<Key>::new();

        for item in keys {
            key_set.insert(item.clone());
        }

        Tree { keys: key_set }
    }

    pub fn linearize_keys(mut keys: Vec<Key>) -> Vec<Key> {

        // To linearize the tree we first sort it.
        keys.sort();

        let nkeys = keys.len();

        // Then we remove the ancestors.
        let mut new_keys = Vec::<Key>::with_capacity(keys.len());

        // Now check pairwise for ancestor relationship and only add to new vector if item
        // is not an ancestor of the next item. Add final element.
        keys.into_iter().enumerate().tuple_windows::<((_, _), (_, _))>().for_each(|((_, a), (j, b))| {
            if !a.is_ancestor(&b) {
                new_keys.push(a.clone());
            }
            if j == (nkeys -1) {
                new_keys.push(b.clone());
            }
        });

        new_keys
    }

    pub fn linearize(&self) -> LinearTree {
        let keys: Vec<Key> = self.keys.iter().copied().collect::<Vec<Key>>();

        LinearTree { keys: Tree::linearize_keys(keys) }
    }
}

impl LinearTree {

    pub fn linearize(&self) -> LinearTree {
        // To linearize the tree we first sort it.

        let mut keys: Vec<Key> = self.keys.iter().copied().collect::<Vec<Key>>();
        keys.sort();

        let nkeys = self.keys.len();

        // Then we remove the ancestors.
        let mut new_keys = Vec::<Key>::with_capacity(self.keys.len());

        // Now check pairwise for ancestor relationship and only add to new vector if item
        // is not an ancestor of the next item. Add final element.
        keys.into_iter().enumerate().tuple_windows::<((_, _), (_, _))>().for_each(|((_, a), (j, b))| {
            if !a.is_ancestor(&b) {
                new_keys.push(a.clone());
            }
            if j == (nkeys -1) {
                new_keys.push(b.clone());
            }
        });

        LinearTree { keys: new_keys }
    }

    pub fn complete_region(a: &Key, b: &Key) -> Vec<Key> {
        // let mut region = Vec::<Key>::new();
        // let mut work_set = a.finest_ancestor(&b).children();

        let a_ancestors: HashSet<Key> = a.ancestors();
        let b_ancestors: HashSet<Key> = b.ancestors();

        let mut working_list: HashSet<Key> = a.finest_ancestor(&b).children().into_iter().collect();

        let mut minimal_tree: Vec<Key> = Vec::new();

        loop {
            let mut aux_list: HashSet<Key> = HashSet::new();
            let mut len = 0;

            for w in &working_list {
                if ((a < w) & (w < b)) & !b_ancestors.contains(w) {
                    aux_list.insert(*w);
                    len += 1;
                } else if a_ancestors.contains(w) | b_ancestors.contains(w) {
                    for child in w.children() {
                        aux_list.insert(child);
                    }
                }
            }

            if len == working_list.len() {
                minimal_tree = aux_list.into_iter().collect();
                break;
            } else {
                working_list = aux_list;
            }
        }

        minimal_tree.sort();
        minimal_tree

    //     while work_set.len() > 0 {
    //         let current_item = work_set.pop().unwrap();
    //         if (current_item > *a) & (current_item < *b) & !b_ancestors.contains(&current_item)
    //         {
    //             region.push(current_item);
    //         } else if (a_ancestors.contains(&current_item))
    //             | (b_ancestors.contains(&current_item))
    //         {
    //             let mut children = current_item.children();
    //             work_set.append(&mut children);
    //         }
    //     }
    //     region.sort();
    //     Tree::linearize_keys(region)
    }

    pub fn complete(&self) -> CompleteLinearTree {
        let a = self.keys.iter().min().unwrap();
        let b = self.keys.iter().max().unwrap();
        let mut completion = LinearTree::complete_region(&a, &b);
        completion.push(a.clone());
        completion.push(b.clone());
        CompleteLinearTree{keys: completion}
    }
}

impl CompleteLinearTree {
    pub fn compute_interior_weights(
        &self,
        root: &Key,
        weights: &Vec<f64>,
    ) -> HashMap<Key, f64> {
        assert!(
            self.keys.len() == weights.len(),
            "Keys and weights must have the same length."
        );

        assert!(
            root.is_ancestor(self.keys.first().unwrap())
                && root.is_ancestor(self.keys.last().unwrap()),
            "`root` is not ancestor of the keys."
        );

        let mut weights_map = HashMap::<Key, f64>::new();

        // Traverse tree bottom up to compute all weights

        for (key, mut weight) in self.keys.iter().copied().zip(weights.iter().copied()) {
            weights_map.insert(key, weight);

            while key != *root {
                let parent = key.parent();

                if let Some(parent_weight) = weights_map.get_mut(&parent) {
                    *parent_weight += weight;
                    weight = *parent_weight;
                } else {
                    weights_map.insert(parent, weight);
                }
            }
        }

        weights_map
    }

    pub fn coarsen_by_weights(
        &self,
        root: &Key,
        weights: &Vec<f64>,
        max_weight: f64,
    ) -> CompleteLinearTree {
        fn coarsen_impl(
            key: &Key,
            weights: &HashMap<Key, f64>,
            result_keys: &mut Vec<Key>,
            max_weight: f64,
        ) {
            if key.level() == DEEPEST_LEVEL {
                // We are at deepest level. Have to add key.
                result_keys.push(key.clone());
            } else if *weights.get(key).unwrap() <= max_weight {
                // Key is below threshold. Also add it.
                result_keys.push(key.clone());
            } else {
                // Key is above threshold. Check if children are in tree.
                if weights.contains_key(&key.first_child()) {
                    // Children are in tree. Therefore iterate through children.
                    for child in key.children() {
                        coarsen_impl(&child, weights, result_keys, max_weight);
                    }
                } else {
                    // Children not in tree. Have to add key itself despite being too big.
                    result_keys.push(key.clone());
                }
            }
        }

        let weights_map = self.compute_interior_weights(&root, &weights);
        let mut result_keys = Vec::<Key>::with_capacity(self.keys.len());
        coarsen_impl(root, &weights_map, &mut result_keys, max_weight);
        result_keys.sort();

        CompleteLinearTree { keys: result_keys }
    }
}

// use ndarray::{Array1, ArrayView1, ArrayView2, Axis};
// use rusty_kernel_tools::RealType;
// use std::collections::{HashMap, HashSet};
// use std::fmt;
// use std::time::Duration;

// pub mod adaptive_octree;
// pub mod regular_octree;

// pub use adaptive_octree::*;
// pub use regular_octree::*;

// /// The type of the octree.
// pub enum OctreeType {
//     /// Regular octree.
//     Regular,
//     /// Use for balanced adaptive octrees.
//     BalancedAdaptive,
//     /// Use for unbalanced adaptive octrees.
//     UnbalancedAdaptive,
// }

// /// The basic Octree data structure
// pub struct Octree<'a, T: RealType> {
//     /// A (3, N) array of N particles.
//     pub particles: ArrayView2<'a, T>,

//     /// The maximum level in the tree.
//     pub max_level: usize,

//     /// The origin of the bounding box for the particles.
//     pub origin: [f64; 3],

//     /// The diameter across each dimension of the bounding box.
//     pub diameter: [f64; 3],

//     /// The non-empty keys for each level of the tree.
//     pub level_keys: HashMap<usize, HashSet<usize>>,

//     /// The keys associated with the particles.
//     pub particle_keys: Array1<usize>,

//     /// The set of near-field keys for each non-empty key.
//     pub near_field: HashMap<usize, HashSet<usize>>,

//     /// The set of keys in the interaction list for each non-empty key.
//     pub interaction_list: HashMap<usize, HashSet<usize>>,

//     /// The index set of particles associated with each leaf key.
//     pub leaf_key_to_particles: HashMap<usize, HashSet<usize>>,

//     /// The set of all non-empty keys in the tree.
//     pub all_keys: HashSet<usize>,

//     /// The type of the Octree.
//     pub octree_type: OctreeType,

//     /// Statistics for the tree.
//     pub statistics: Statistics,
// }

// /// A structure that stores various statistics for a tree.
// pub struct Statistics {
//     pub number_of_particles: usize,
//     pub max_level: usize,
//     pub number_of_leafs: usize,
//     pub number_of_keys: usize,
//     pub creation_time: Duration,
//     pub minimum_number_of_particles_in_leaf: usize,
//     pub maximum_number_of_particles_in_leaf: usize,
//     pub average_number_of_particles_in_leaf: f64,
// }

// impl std::fmt::Display for Statistics {
//     /// Create an output string for tree statistics.
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         write!(
//             f,
//             "\n\nOctree Statistics\n\
//                    ==============================\n\
//                    Number of Particles: {}\n\
//                    Maximum level: {}\n\
//                    Number of leaf keys: {}\n\
//                    Number of keys in tree: {}\n\
//                    Creation time [s]: {}\n\
//                    Minimum number of particles in leaf node: {}\n\
//                    Maximum number of particles in leaf node: {}\n\
//                    Average number of particles in leaf node: {:.2}\n\
//                    ==============================\n\n
//                    ",
//             self.number_of_particles,
//             self.max_level,
//             self.number_of_leafs,
//             self.number_of_keys,
//             (self.creation_time.as_millis() as f64) / 1000.0,
//             self.minimum_number_of_particles_in_leaf,
//             self.maximum_number_of_particles_in_leaf,
//             self.average_number_of_particles_in_leaf
//         )
//     }
// }

// /// Given the set of all keys, compute the interaction list for each key.
// ///
// /// Returns a map from keys to the corresponding interaction list, represented by
// /// a set of keys.
// pub fn compute_interaction_list_map(all_keys: &HashSet<usize>) -> HashMap<usize, HashSet<usize>> {
//     use crate::morton::compute_interaction_list;
//     use rayon::prelude::*;

//     let mut interaction_list_map = HashMap::<usize, HashSet<usize>>::new();

//     for &key in all_keys {
//         interaction_list_map.insert(key, HashSet::<usize>::new());
//     }

//     interaction_list_map
//         .par_iter_mut()
//         .for_each(|(&key, hash_set)| {
//             let current_interaction_list = compute_interaction_list(key);
//             hash_set.extend(&current_interaction_list);
//         });

//     interaction_list_map
// }

// /// Given the set of all keys, compute the near field for each key.
// ///
// /// Returns a map from keys to the corresponding near field, represented by
// /// a set of keys.
// pub fn compute_near_field_map(all_keys: &HashSet<usize>) -> HashMap<usize, HashSet<usize>> {
//     use crate::morton::compute_near_field;
//     use rayon::prelude::*;

//     let mut near_field_map = HashMap::<usize, HashSet<usize>>::new();

//     for &key in all_keys {
//         near_field_map.insert(key, HashSet::<usize>::new());
//     }

//     near_field_map.par_iter_mut().for_each(|(&key, hash_set)| {
//         let current_near_field = compute_near_field(key);
//         hash_set.extend(&current_near_field);
//     });

//     near_field_map
// }

// /// Compute the leaf map.
// ///
// /// Returns a map from leaf keys to associated particle indices.
// pub fn compute_leaf_map(particle_keys: ArrayView1<usize>) -> HashMap<usize, HashSet<usize>> {
//     use itertools::Itertools;

//     let mut leaf_key_to_particles = HashMap::<usize, HashSet<usize>>::new();
//     for &key in particle_keys.iter().unique() {
//         leaf_key_to_particles.insert(key, HashSet::<usize>::new());
//     }

//     for (index, key) in particle_keys.iter().enumerate() {
//         leaf_key_to_particles.get_mut(key).unwrap().insert(index);
//     }

//     leaf_key_to_particles
// }

// /// Given an array of keys. Return the level information of the tree.
// ///
// /// The function returns a 3-tuple `(max_level, all_keys, level_keys)`.
// /// `max_level` us a `usize` that contains the maximum level of the keys.
// /// The set `all_keys` contains all keys from the tree by completing the tree
// /// from the leaf onwards to the top and storing all parent keys along the way.
// /// The map `level_keys` is a map from a given level to the set of all keys contained
// /// in the level.
// pub fn compute_level_information(
//     particle_keys: ArrayView1<usize>,
// ) -> (usize, HashSet<usize>, HashMap<usize, HashSet<usize>>) {
//     use crate::morton::{find_level, find_parent};
//     use std::iter::FromIterator;

//     let max_level = particle_keys
//         .iter()
//         .map(|&item| find_level(item))
//         .max()
//         .unwrap();

//     let nlevels = max_level + 1;
//     let leaf_keys: HashSet<usize> = particle_keys.iter().copied().collect();
//     let mut level_keys = HashMap::<usize, HashSet<usize>>::new();

//     // Distribute the keys among different sets for each level
//     for current_level in 0..nlevels {
//         level_keys.insert(
//             current_level,
//             HashSet::from_iter(
//                 leaf_keys
//                     .iter()
//                     .filter(|&&key| find_level(key) == current_level)
//                     .copied(),
//             ),
//         );
//     }

//     // Now fill up the sets with all the various parent keys.
//     for current_level in (1..nlevels).rev() {
//         let parent_index = current_level - 1;
//         let current_set: HashSet<usize> = level_keys
//             .get(&current_level)
//             .unwrap()
//             .iter()
//             .copied()
//             .collect();
//         let parent_set = level_keys.get_mut(&parent_index).unwrap();
//         parent_set.extend(current_set.iter().map(|&key| find_parent(key)));
//     }

//     let mut all_keys = HashSet::<usize>::new();

//     for (_, current_keys) in level_keys.iter() {
//         all_keys.extend(current_keys.iter());
//     }

//     (max_level, all_keys, level_keys)
// }

// /// Create a regular octree from an array of particles.
// ///
// /// Returns the set of all non-empty keys associated with a regular octree created
// /// from an array of particles.
// /// # Arguments
// /// * `particles` - A (3, N) array of particles
// /// * `max_level` - The deepest level of the tree.
// /// * `origin` - The origin of the bounding box.
// /// * `diameter` - The diameter of the bounding box in each dimension.
// pub fn compute_complete_regular_tree<T: RealType>(
//     particles: ArrayView2<T>,
//     max_level: usize,
//     origin: &[f64; 3],
//     diameter: &[f64; 3],
// ) -> HashSet<usize> {
//     use super::morton::encode_points;

//     let keys = encode_points(particles, max_level, origin, diameter);
//     let (_, all_keys, _) = compute_level_information(keys.view());

//     all_keys
// }

// /// Export an octree to vtk
// pub fn export_to_vtk<T: RealType>(tree: &Octree<T>, filename: &str) {
//     use super::morton::{serialize_box_from_key};
//     use std::iter::FromIterator;
//     use vtkio::model::*;
//     use std::path::PathBuf;

//     let filename = filename.to_owned();

//     // We use a vtk voxel type, which has
//     // 8 points per cell, i.e. 24 float numbers
//     // per cell.
//     const POINTS_PER_CELL: usize = 8;

//     let num_particles = tree.particles.len_of(Axis(1));

//     let all_keys = &tree.all_keys;
//     let num_keys = all_keys.len();

//     let num_floats = 3 * POINTS_PER_CELL * num_keys;

//     let mut cell_points = Vec::<f64>::with_capacity(num_floats);

//     for &key in all_keys {
//         let serialized = serialize_box_from_key(key, &tree.origin, &tree.diameter);
//         cell_points.extend(serialized);
//     }

//     for index in 0..num_particles {
//         cell_points.push(tree.particles[[0, index]].to_f64().unwrap());
//         cell_points.push(tree.particles[[1, index]].to_f64().unwrap());
//         cell_points.push(tree.particles[[2, index]].to_f64().unwrap());
//     }

//     let num_points = 8 * (num_keys as u64) + (num_particles as u64);

//     let connectivity = Vec::<u64>::from_iter(0..num_points);
//     let mut offsets = Vec::<u64>::from_iter((0..(num_keys as u64)).map(|item| 8 * item + 8));
//     offsets.push(num_points);

//     let mut types = vec![CellType::Voxel; num_keys];
//     types.push(CellType::PolyVertex);

//     let mut cell_data = Vec::<i32>::with_capacity(num_points as usize);

//     for _ in 0..num_keys {
//         cell_data.push(0);
//     }
//     cell_data.push(1);


//     let model = Vtk {
//         version: Version { major: 1, minor: 0 },
//         title: String::new(),
//         byte_order: ByteOrder::BigEndian,
//         file_path: Some(PathBuf::from(&filename)),
//         data: DataSet::inline(UnstructuredGridPiece {
//             points: IOBuffer::F64(cell_points),
//             cells: Cells {
//                 cell_verts: VertexNumbers::XML {
//                     connectivity: connectivity,
//                     offsets: offsets,
//                 },
//                 types: types,
//             },
//             data: Attributes {
//                 point: vec![],
//                 cell: vec![Attribute::DataArray(DataArrayBase {
//                     name: String::from("colors"),
//                     elem: ElementType::Scalars {
//                         num_comp: 1,
//                         lookup_table: None,
//                     },
//                     data: IOBuffer::I32(cell_data),
//                 })],
//             },
//         }),
//     };


//     model.export(filename).unwrap();
// }
