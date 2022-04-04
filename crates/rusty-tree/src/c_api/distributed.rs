use mpi::{
    environment::Universe,
    ffi::MPI_Comm,
    topology::{Color, UserCommunicator, SystemCommunicator},
    traits::*
};

use crate::{
    distributed::DistributedTree,
    types::{
        domain::Domain,
        point::{PointType, Point},
        morton::MortonKey
    }
};

use rand::prelude::*;
use rand::SeedableRng;

use std::ptr;

#[no_mangle]
pub extern "C" fn distributed_tree_from_points(
    p_points: *const [PointType; 3],
    npoints: usize,
    balanced: bool,
    comm: *mut usize
) -> *mut DistributedTree {
    let points = unsafe { std::slice::from_raw_parts(p_points, npoints) }; 
    let mut comm = std::mem::ManuallyDrop::new(unsafe {UserCommunicator::from_raw(*comm as MPI_Comm)}.unwrap());
    Box::into_raw(Box::new(DistributedTree::new(points, balanced, &mut comm)))
}

#[no_mangle]
pub extern "C" fn distributed_tree_nkeys(
    p_tree: *const DistributedTree,
) -> usize {
    let mut tree = unsafe { &*p_tree };
    tree.keys.len()
}

#[no_mangle]
pub extern "C" fn distributed_tree_npoints(
    p_tree: *const DistributedTree,
) -> usize {
    let mut tree = unsafe { &*p_tree };
    tree.points.len()
}

#[no_mangle]
pub extern "C" fn distributed_tree_keys(
    p_tree: *const DistributedTree,
)  ->  *const MortonKey {
    let tree = unsafe { &*p_tree };
    tree.keys.as_ptr()
}

#[no_mangle]
pub extern "C" fn distributed_tree_points(
    p_tree: *const DistributedTree,
)  ->  *const Point {
    let tree = unsafe { &*p_tree };
    tree.points.as_ptr()
}

#[no_mangle]
pub extern "C" fn distributed_tree_balanced(
    p_tree: *const DistributedTree,
) -> bool {
    let tree = unsafe { &*p_tree };
    tree.balanced
}
