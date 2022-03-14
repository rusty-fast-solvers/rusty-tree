// //! Data structures and functions to create regular and adaptive Octrees.

use std::{
    ops::Deref,
    collections::{HashMap, HashSet}
};

use itertools::Itertools;

use crate::{
    constants::DEEPEST_LEVEL,
    types::morton::{MortonKey, KeyType}
};


#[derive(Debug)]
pub struct Tree {
    pub keys: Vec<MortonKey>,
}

impl Tree {

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

    pub fn complete(self: &mut Tree) {
        let a = self.keys.iter().min().unwrap();
        let b = self.keys.iter().max().unwrap();
        let mut completion = Tree::complete_region(&a, &b);
        completion.push(a.clone());
        completion.push(b.clone());
        self.keys = completion;
    }

    pub fn linearize(self: &mut Tree) {
        self.keys = Tree::linearize_keys(self.keys.clone());
    }

    pub fn sort(self: &mut Tree) {
        self.keys.sort();
    }
}

impl Deref for Tree {
    type Target = Vec<MortonKey>;

    fn deref(&self) -> &Self::Target {
        &self.keys
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    use rand::prelude::*;
    use rand::{SeedableRng};

    use crate::octree::Tree;
    use crate::types::{
        morton::MortonKey,
        domain::Domain,
        point::Point,
    };

    /// Tree fixture
    fn tree () -> Tree {
        let ncrit: usize = 150;
        let npoints: u64 = 10000;

        let domain = Domain{
            origin: [0., 0., 0.],
            diameter: [1., 1., 1.]
        };

        let mut range = StdRng::seed_from_u64(0);
        let between = rand::distributions::Uniform::from(0.0..1.0);
        let mut points = Vec::new();

        for _ in 0..npoints {
            points.push([between.sample(&mut range), between.sample(&mut range), between.sample(&mut range)])
        }

        let mut points: Vec<Point> = points
        .iter()
        .map(|p| Point{coordinate: p.clone(), global_idx: 0, key: MortonKey::from_point(&p, &domain)})
        .collect();
        
        let keys: Vec<MortonKey> = points
        .iter()
        .map(|p| p.key)
        .collect();

        Tree {keys}
    }

    #[test]
    fn test_sort() {
        let t = tree();
        println!("{:?}", t.iter().len());
        assert!(false);
    }

    #[test]
    fn test_linearize() {
        
    }

    #[test]
    fn test_complete() {
        assert!(true);
    }

}
