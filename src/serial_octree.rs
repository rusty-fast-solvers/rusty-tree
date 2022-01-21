//! Algorithms for serial Octrees

use crate::morton::MortonKey;
use crate::types::{Domain, KeyType, Point, Points};
use std::collections::{HashMap, HashSet};

pub enum NodeType {
    InteriorNode,
    LeafNode((usize, usize)),
}

pub struct Tree {
    nodes: HashMap<MortonKey, NodeType>,
    points: Points,
    domain: Domain,
}

impl Tree {
    pub fn new(points: Points, domain: Domain) -> Self {
        let mut nodes = HashMap::<MortonKey, NodeType>::new();

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
            tree.nodes.insert(key, NodeType::LeafNode((index, index + 1)));

            while current_key.level() > 0 {
                current_key = current_key.parent();
                if tree.nodes.contains_key(&current_key) {
                    break;
                }
                else {
                    // Insert the key
                    tree.nodes.insert(current_key, NodeType::InteriorNode);
                }
            }
        }

        tree
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
}
