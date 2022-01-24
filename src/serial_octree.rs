//! Algorithms for serial Octrees

use crate::morton::MortonKey;
use crate::types::{Domain, Point, Points};
use std::collections::{HashMap, HashSet};

pub enum NodeType {
    InteriorNode(HashSet<MortonKey>),
    LeafNode,
}

pub struct NodeEntry {
    node_type: NodeType,
    points: HashSet<usize>,
}

pub struct Tree {
    nodes: HashMap<MortonKey, NodeEntry>,
    points: Points,
    domain: Domain,
}

impl Tree {
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
}
