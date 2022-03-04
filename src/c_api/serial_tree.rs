//! C API for serial trees

use crate::types::KeyType;
use crate::morton::MortonKey;
use crate::serial_octree::{Tree, LinearTree};
use std::slice::from_raw_parts;


#[no_mangle]
pub extern "C" fn tree_from_morton_keys(data: *const KeyType, len: usize) -> *mut Tree {

    let slice = unsafe {from_raw_parts(data, len) };
    let iter = slice.iter().map(|&item| MortonKey::from_morton(item));
    let tree = Tree::from_iterable(iter);

    Box::into_raw(Box::new(tree))

}

#[no_mangle]
pub extern "C" fn tree_linearize(tree: *mut Tree) -> *mut LinearTree {


    let linear_tree = unsafe {(*tree).linearize() };
    Box::into_raw(Box::new(linear_tree))

}
