//! Algorithms for serial Octrees

use crate::morton::MortonKey;
use crate::types::{Domain, KeyType, Point, Points};

pub fn points_to_sorted_morton_keys(points: &Points, domain: &Domain) -> (Points, Vec<KeyType>) {
    let mut indices: Vec<usize> = (0..points.len()).collect();
    let morton_keys: Vec<KeyType> = points
        .iter()
        .map(|&point| MortonKey::from_point(&point.coord, domain).morton())
        .collect();

    indices.sort_unstable_by_key(|&index| morton_keys[index]);

    let mut sorted_keys: Vec<KeyType> = Vec::new();
    let mut sorted_points: Vec<Point> = Vec::new();

    for index in indices {
        sorted_keys.push(morton_keys[index]);
        sorted_points.push(points[index]);
    }

    (sorted_points, sorted_keys)
}

pub fn adaptive_tree(points: &Points, morton_key)
