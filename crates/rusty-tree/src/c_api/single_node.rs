//! C API for trees on a single node

use crate::{
    types::{
        morton::{KeyType, MortonKey},
    },
    single_node::Tree
};
use std::slice::from_raw_parts;


#[no_mangle]
pub extern "C" fn tree_from_morton_keys(data: *const KeyType, len: usize) -> *mut Tree {
    let slice = unsafe {from_raw_parts(data, len) };
    let keys = slice.iter().map(|&item| MortonKey::from_morton(item)).collect();
    let tree = Tree{keys};
    Box::into_raw(Box::new(tree))
}

// #[no_mangle]
// pub extern "C" fn tree_linearize_keys(keys: *const Vec<MortonKey>) -> *mut Vec<MortonKey> {
//     let linearized = unsafe {Tree::linearize_keys(*keys.clon) };
//     Box::into_raw(Box::new(linearized))
// }

#[no_mangle]
pub extern "C" fn tree_complete_region(a: *const MortonKey, b: *const MortonKey) -> *mut Vec<MortonKey> {
    let completed = unsafe {Tree::complete_region(&*a, &*b) };
    Box::into_raw(Box::new(completed))
}

// #[no_mangle]
// pub extern "C" fn balance(tree: *mut Tree) -> *mut Tree {
//     let balanced = tree.balance();
//     Box::into_raw(Box::new(balanced))
// }