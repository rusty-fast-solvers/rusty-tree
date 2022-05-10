//! Wrappers for Distributed Tree interface

use mpi::{environment::Universe, ffi::MPI_Comm, topology::UserCommunicator, traits::*};

use crate::{
    distributed::DistributedTree,
    types::{
        domain::Domain,
        morton::MortonKey,
        point::{Point, PointType},
    },
};

use rand::prelude::*;
use rand::SeedableRng;

#[no_mangle]
pub extern "C" fn distributed_tree_from_points(
    p_points: *const [PointType; 3],
    npoints: usize,
    balanced: bool,
    world: *mut usize,
) -> *mut DistributedTree {
    let points = unsafe { std::slice::from_raw_parts(p_points, npoints) };
    let world = std::mem::ManuallyDrop::new(unsafe {
        UserCommunicator::from_raw(*(world as *const MPI_Comm)).unwrap()
    });
    Box::into_raw(Box::new(DistributedTree::new(points, balanced, &world)))
}

#[no_mangle]
pub extern "C" fn distributed_tree_nkeys(p_tree: *const DistributedTree) -> usize {
    let tree = unsafe { &*p_tree };
    tree.keys.len()
}

#[no_mangle]
pub extern "C" fn distributed_tree_npoints(p_tree: *const DistributedTree) -> usize {
    let tree = unsafe { &*p_tree };
    tree.points.len()
}

#[no_mangle]
pub extern "C" fn distributed_tree_keys(p_tree: *const DistributedTree) -> *const MortonKey {
    let tree = unsafe { &*p_tree };
    tree.keys.as_ptr()
}

#[no_mangle]
pub extern "C" fn distributed_tree_points(p_tree: *const DistributedTree) -> *const Point {
    let tree = unsafe { &*p_tree };
    tree.points.as_ptr()
}

#[no_mangle]
pub extern "C" fn distributed_tree_balanced(p_tree: *const DistributedTree) -> bool {
    let tree = unsafe { &*p_tree };
    tree.balanced
}
