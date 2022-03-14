// //! Data structures and functions to create regular and adaptive Octrees.

//! Algorithms for serial Octrees

use crate::{
    constants::DEEPEST_LEVEL,
    types::morton::{MortonKey, KeyType}
};

use itertools::Itertools;
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
pub struct Tree {
    pub keys: Vec<MortonKey>,
}

impl Tree {
    pub fn from_iterable<T: Iterator<Item = MortonKey>>(keys: T) -> Tree {

        let mut key_set = HashSet::<MortonKey>::new();

        for item in keys {
            key_set.insert(item.clone());
        }

        let keys: Vec<MortonKey> = key_set.into_iter().collect();

        Tree { keys }
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



    pub fn complete_region(a: &MortonKey, b: &MortonKey) -> Vec<MortonKey> {
        // let mut region = Vec::<Key>::new();
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
    }

    pub fn complete(&self) -> Tree {
        let a = self.keys.iter().min().unwrap();
        let b = self.keys.iter().max().unwrap();
        let mut completion = Tree::complete_region(&a, &b);
        completion.push(a.clone());
        completion.push(b.clone());
        Tree {keys: completion}
    }

    pub fn linearize(&self) -> Tree {
        let keys: Vec<MortonKey> = self.keys.iter().copied().collect::<Vec<MortonKey>>();

        Tree { keys: Tree::linearize_keys(keys) }
    }

}
