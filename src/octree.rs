//! Octree interface

use crate::types::Key;
use crate::constants::DEEPEST_LEVEL;
use itertools::Itertools;
use std::collections::{HashMap, HashSet};

pub enum NodeType {
    InteriorNode(HashSet<Key>),
    LeafNode,
}

pub struct Tree {
    keys: HashSet<Key>,
}

pub struct LinearTree {
    keys: Vec<Key>,
}

pub struct CompleteLinearTree {
    keys: Vec<Key>,
}

impl Tree {
    pub fn from_iterable<T: Iterator<Item = Key>>(keys: T) -> Tree {
        let mut key_set = HashSet::<Key>::new();
        for item in keys {
            key_set.insert(item);
        }

        Tree { keys: key_set }
    }

    pub fn from_hash_set(keys: HashSet<Key>) -> Tree {
        Tree { keys }
    }

    pub fn linearize(&self) -> LinearTree {
        // To linearize the tree we first sort it.

        let mut keys: Vec<Key> = self.keys.iter().copied().collect::<Vec<Key>>();
        keys.sort();

        // Then we remove the ancestors.

        let mut new_keys = Vec::<Key>::with_capacity(keys.len());

        // Now check pairwise for ancestor relationship and only add to new vector if item
        // is not an ancestor of the next item.

        keys.into_iter().tuple_windows::<(_, _)>().for_each(|item| {
            if !item.0.is_ancestor(&item.1) {
                new_keys.push(item.0);
            }
        });

        LinearTree { keys: new_keys }
    }
}

impl LinearTree {
    pub fn complete(&mut self, root: Key) -> CompleteLinearTree {
        fn complete_region(a: &Key, b: &Key) -> Vec<Key> {
            let mut region = Vec::<Key>::new();
            let mut work_set = a.finest_ancestor(&b).children();

            let a_ancestors = a.ancestors();
            let b_ancestors = b.ancestors();

            while work_set.len() > 0 {
                let current_item = work_set.pop().unwrap();
                if (current_item > *a) & (current_item < *b) & !b_ancestors.contains(&current_item)
                {
                    region.push(current_item);
                } else if (a_ancestors.contains(&current_item))
                    | (b_ancestors.contains(&current_item))
                {
                    let mut children = current_item.children();
                    work_set.append(&mut children);
                }
            }

            region.sort();
            region
        }

        assert!(
            root.is_ancestor(self.keys.first().unwrap())
                && root.is_ancestor(self.keys.last().unwrap()),
            "`root` is not ancestor of the keys."
        );

        let finest_first_child = root.finest_first_child();
        let finest_last_child = root.finest_last_child();

        if *self.keys.first().unwrap() != finest_first_child {
            self.keys.insert(0, finest_first_child);
        }
        if *self.keys.last().unwrap() != finest_last_child {
            self.keys.push(finest_last_child);
        }

        let mut new_keys = Vec::<Key>::new();

        for (first, second) in self.keys.iter().tuple_windows::<(_, _)>() {
            let region = complete_region(first, second);
            new_keys.push(first.clone());
            new_keys.extend(region.iter());
            new_keys.push(second.clone());
        }

        CompleteLinearTree { keys: new_keys }
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

