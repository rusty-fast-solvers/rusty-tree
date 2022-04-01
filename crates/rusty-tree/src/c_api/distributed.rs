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
        point::PointType,
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
pub extern "C" fn distributed_tree_random(
    balanced: bool,
    npoints: usize,
    comm: *mut usize
) -> *mut DistributedTree {

    let mut range = StdRng::seed_from_u64(0);
    let between = rand::distributions::Uniform::<f64>::from(0.0..1.0);
    let mut points = Vec::new();

    for _ in 0..npoints {
        points.push([
            between.sample(&mut range),
            between.sample(&mut range),
            between.sample(&mut range),
        ])
    };

    let mut comm = std::mem::ManuallyDrop::new(unsafe {UserCommunicator::from_raw(*comm as MPI_Comm)}.unwrap());
    Box::into_raw(Box::new(DistributedTree::new(&points, balanced, &mut comm)))
}

#[no_mangle]
pub extern "C" fn distributed_tree_n_keys(
    p_tree: *const DistributedTree,
) -> usize {
    let mut tree = unsafe { &*p_tree };
    tree.keys.len()
}

#[no_mangle]
pub extern "C" fn distributed_tree_n_points(
    p_tree: *const DistributedTree,
) -> usize {
    let mut tree = unsafe { &*p_tree };
    tree.points.len()
}

#[no_mangle]
pub extern "C" fn distributed_tree_keys_slice(
    p_tree: *const DistributedTree,
    ptr: *mut usize,
    lidx: usize,
    ridx: usize
) {
    let tree = unsafe { &*p_tree };
    let nkeys = ridx-lidx;

    let boxes = unsafe {std::slice::from_raw_parts_mut(ptr, nkeys)};

    for i in 0..nkeys {
        let j = i+lidx;
        let key = tree.keys[j].clone();
        boxes[i] = Box::into_raw(Box::new(key)) as usize;
    }
}

#[no_mangle]
pub extern "C" fn distributed_tree_points_slice(
    p_tree: *const DistributedTree,
    ptr: *mut usize,
    lidx: usize,
    ridx: usize
) {
    let tree = unsafe { &*p_tree };
    let npoints = ridx-lidx;

    let boxes = unsafe {std::slice::from_raw_parts_mut(ptr, npoints)};

    for i in 0..npoints {
        let j = i+lidx;
        let point = tree.points[j].clone();
        boxes[i] = Box::into_raw(Box::new(point)) as usize;
    }
}

#[no_mangle]
pub extern "C" fn distributed_tree_balanced(
    p_tree: *const DistributedTree,
) -> bool {
    let tree = unsafe { &*p_tree };
    tree.balanced
}
