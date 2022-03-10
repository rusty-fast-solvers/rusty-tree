//! Algorithms for serial Octrees

use crate::morton::MortonKey;
// use crate::types::{Domain, KeyType, Point, Points};
use crate::DEEPEST_LEVEL;
use itertools::Itertools;
use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;

pub enum NodeType {
    InteriorNode(HashSet<MortonKey>),
    LeafNode,
}

#[derive(Debug)]
pub struct Tree {
    pub keys: HashSet<MortonKey>,
}

#[derive(Debug)]
pub struct LinearTree {
    pub keys: Vec<MortonKey>,
}

#[derive(Debug)]
pub struct CompleteLinearTree {
    pub keys: Vec<MortonKey>,
}

impl Tree {
    pub fn from_iterable<T: Iterator<Item = MortonKey>>(keys: T) -> Tree {

        let mut key_set = HashSet::<MortonKey>::new();

        for item in keys {
            key_set.insert(item.clone());
        }

        Tree { keys: key_set }
    }

    pub fn linearize_keys(mut keys: Vec<MortonKey>) -> Vec<MortonKey> {

        // To linearize the tree we first sort it.
        keys.sort();

        let nkeys = keys.len();

        // Then we remove the ancestors.
        let mut new_keys = Vec::<MortonKey>::with_capacity(keys.len());

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
        let keys: Vec<MortonKey> = self.keys.iter().copied().collect::<Vec<MortonKey>>();

        LinearTree { keys: Tree::linearize_keys(keys) }
    }
}

impl LinearTree {

    pub fn linearize(&self) -> LinearTree {
        // To linearize the tree we first sort it.

        let mut keys: Vec<MortonKey> = self.keys.iter().copied().collect::<Vec<MortonKey>>();
        keys.sort();

        let nkeys = self.keys.len();

        // Then we remove the ancestors.
        let mut new_keys = Vec::<MortonKey>::with_capacity(self.keys.len());

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

    pub fn complete_region(a: &MortonKey, b: &MortonKey) -> Vec<MortonKey> {
        // let mut region = Vec::<MortonKey>::new();
        // let mut work_set = a.finest_ancestor(&b).children();

        let a_ancestors: HashSet<MortonKey> = a.ancestors();
        let b_ancestors: HashSet<MortonKey> = b.ancestors();

        let mut working_list: HashSet<MortonKey> = a.finest_ancestor(&b).children().into_iter().collect();

        let mut minimal_tree: Vec<MortonKey> = Vec::new();

        loop {
            let mut aux_list: HashSet<MortonKey> = HashSet::new();
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
        root: &MortonKey,
        weights: &Vec<f64>,
    ) -> HashMap<MortonKey, f64> {
        assert!(
            self.keys.len() == weights.len(),
            "Keys and weights must have the same length."
        );

        assert!(
            root.is_ancestor(self.keys.first().unwrap())
                && root.is_ancestor(self.keys.last().unwrap()),
            "`root` is not ancestor of the keys."
        );

        let mut weights_map = HashMap::<MortonKey, f64>::new();

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
        root: &MortonKey,
        weights: &Vec<f64>,
        max_weight: f64,
    ) -> CompleteLinearTree {
        fn coarsen_impl(
            key: &MortonKey,
            weights: &HashMap<MortonKey, f64>,
            result_keys: &mut Vec<MortonKey>,
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
        let mut result_keys = Vec::<MortonKey>::with_capacity(self.keys.len());
        coarsen_impl(root, &weights_map, &mut result_keys, max_weight);
        result_keys.sort();

        CompleteLinearTree { keys: result_keys }
    }
}

/* impl Tree {
    pub fn new(points: Points, domain: Domain) -> Self {
        let nodes = HashMap::<MortonKey, NodeEntry>::new();

        // First sort the points
        let (sorted_points, sorted_keys) = points_to_sorted_morton_keys(&points, &domain);

        let mut tree = Tree {
            nodes,
            points: sorted_points,
            domain,
        };

        // Insert the keys and their ancestors into the tree

        for (index, &key) in sorted_keys.iter().enumerate() {
            let mut current_key = key.clone();
            tree.nodes.insert(
                key,
                NodeEntry {
                    node_type: NodeType::LeafNode,
                    points: HashSet::from([index]),
                },
            );

            while current_key.level() > 0 {
                let child_key = current_key.clone();
                current_key = current_key.parent();
                if let Some(entry) = tree.nodes.get_mut(&current_key) {
                    if let NodeType::InteriorNode(children) = &mut entry.node_type {
                        children.insert(child_key);
                        entry.points.insert(index);
                    }
                } else {
                    tree.nodes.insert(
                        key,
                        NodeEntry {
                            node_type: NodeType::InteriorNode(HashSet::from([child_key])),
                            points: HashSet::from([index]),
                        },
                    );
                }
            }
        }

        tree
    }

    pub fn remove_node(&mut self, key: &MortonKey) -> Result<(), ()> {
        if let Some(node_entry) = self.nodes.remove(key) {
            if let NodeType::InteriorNode(children) = &node_entry.node_type {
                for child in children {
                    self.remove_node(&child).unwrap();
                }
                if let Some(mut parent_entry) = self.nodes.get_mut(&key.parent()) {
                    if let NodeType::InteriorNode(children) = &mut parent_entry.node_type {
                        children.remove(key);
                        if children.len() == 0 {
                            parent_entry.node_type = NodeType::LeafNode;
                        }
                    }
                }
            }

            Ok(())
        } else {
            Err(())
        }
    }
}

fn points_to_sorted_morton_keys(points: &Points, domain: &Domain) -> (Points, Vec<MortonKey>) {
    let mut indices: Vec<usize> = (0..points.len()).collect();
    let morton_keys: Vec<MortonKey> = points
        .iter()
        .map(|&point| MortonKey::from_point(&point.coord, domain))
        .collect();

    indices.sort_unstable_by_key(|&index| morton_keys[index]);

    let mut sorted_keys: Vec<MortonKey> = Vec::new();
    let mut sorted_points: Vec<Point> = Vec::new();

    for index in indices {
        sorted_keys.push(morton_keys[index]);
        sorted_points.push(points[index]);
    }

    (sorted_points, sorted_keys)
} */
