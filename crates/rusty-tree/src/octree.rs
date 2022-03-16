// //! Data structures and functions to create regular and adaptive Octrees.

use std::{
    ops::{Deref, DerefMut},
    collections::{HashSet}
};

use itertools::Itertools;

use crate::{
    constants::DEEPEST_LEVEL,
    types::morton::{MortonKey}
};


#[derive(Debug)]
pub struct Tree {
    pub keys: Vec<MortonKey>,
}

impl Tree {

    /// Input must be sorted!
     pub fn linearize_keys(keys: Vec<MortonKey>) -> Vec<MortonKey> {

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

        let mut a_ancestors: HashSet<MortonKey> = a.ancestors();
        let mut b_ancestors: HashSet<MortonKey> = b.ancestors();

        a_ancestors.remove(a);
        b_ancestors.remove(b);

        let mut work_list: Vec<MortonKey> = a.finest_ancestor(&b).children().into_iter().collect();

        let mut minimal_tree: Vec<MortonKey> = Vec::new();

        while work_list.len() > 0 {
        // println!("work list {:?} \n", work_list);
            let current_item = work_list.pop().unwrap();
            if (current_item > *a) & (current_item < *b) & !b_ancestors.contains(&current_item)
            {
                minimal_tree.push(current_item);
            } else if (a_ancestors.contains(&current_item)) | (b_ancestors.contains(&current_item))
            {
                let mut children = current_item.children();
                work_list.append(&mut children);
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
        completion.sort();
        self.keys = completion;
    }

    pub fn linearize(self: &mut Tree) {
        self.keys.sort();
        self.keys = Tree::linearize_keys(self.keys.clone());
    }

    pub fn sort(self: &mut Tree) {
        self.keys.sort();
    }

    /// Balance a tree, and remove overlaps
    pub fn balance(&self) -> Tree {

        let mut balanced: HashSet<MortonKey> = self.keys.iter().cloned().collect();

        for level in (0..DEEPEST_LEVEL).rev() {
            let work_list: Vec<MortonKey> = balanced
                .iter()
                .filter(|key| key.level() == level)
                .cloned()
                .collect();

            for key in work_list {
                let neighbors = key.neighbors();

                for neighbor in neighbors {
                    let parent = neighbor.parent();
                    if !balanced.contains(&neighbor) && !balanced.contains(&neighbor) {
                        balanced.insert(parent);

                        if parent.level() > 0 {
                            for sibling in parent.siblings() {
                                balanced.insert(sibling);
                            }
                        }
                    }
                }
            }
        }

        let mut balanced: Vec<MortonKey> = balanced.into_iter().collect();
        balanced.sort();
        let linearized = Tree::linearize_keys(balanced);
        Tree{keys: linearized}
    }
}

impl Deref for Tree {
    type Target = Vec<MortonKey>;

    fn deref(&self) -> &Self::Target {
        &self.keys
    }
}

impl DerefMut for Tree {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.keys
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

    fn tree_fixture () -> Tree {
        let npoints: u64 = 1000;

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

        let points: Vec<Point> = points
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
    fn test_linearize() {
        let mut tree = tree_fixture();
        tree.linearize();

        // Test that a linearized tree is sorted
        for i in 0..(tree.iter().len()-1) {
            let a = tree[i];
            let b = tree[i+1];
            assert!(a <= b);
        }

        // Test that elements in a linearized tree are unique
        let unique: HashSet<MortonKey> = tree.iter().cloned().collect();
        assert!(unique.len() == tree.len());

        // Test that a linearized tree contains no overlaps
        let mut copy: Vec<MortonKey> = tree.keys.iter().cloned().collect();
        for &key in tree.iter() {
            let ancestors = key.ancestors();
            copy.retain(|&k| k != key);

            for ancestor in &ancestors {
                assert!(!copy.contains(ancestor))
            }
        }
    }

    #[test]
    fn test_complete_region() {

        let a: MortonKey = MortonKey { anchor: [0, 0, 0], morton: 16};
        let b: MortonKey = MortonKey {anchor: [65535, 65535, 65535], morton: 0b111111111111111111111111111111111111111111111111000000000010000};

        let region = Tree::complete_region(&a, &b);

        let fa = a.finest_ancestor(&b);

        let min = region.iter().min().unwrap();
        let max = region.iter().max().unwrap();

        // Test that bounds are satisfied
        assert!(a <= *min);
        assert!(b >= *max);

        // Test that FCA is an ancestor of all nodes in the result
        for node in region.iter() {
            let ancestors = node.ancestors();
            assert!(ancestors.contains(&fa));
        }

        // Test that completed region doesn't contain its bounds
        assert!(!region.contains(&a));
        assert!(!region.contains(&b));

        // Test that the compeleted region doesn't contain any overlaps
        for node in region.iter() {
            let mut ancestors = node.ancestors();
            ancestors.remove(node);
            for ancestor in ancestors.iter() {
                assert!(!region.contains(ancestor))
            }
        }

        // Test that the region is sorted
        for i in 0..region.iter().len()-1 {
            let a = region[i];
            let b = region[i+1];

            assert!(a <= b);
        }
    }
}
