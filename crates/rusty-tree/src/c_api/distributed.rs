//! Wrappers for Distributed Tree interface
use mpi::{ffi::MPI_Comm, topology::UserCommunicator, traits::*};
use std::ffi::CString;
use std::os::raw::c_char;

use crate::{
    data::{HDF5, JSON, VTK},
    distributed::DistributedTree,
    types::{
        domain::Domain,
        morton::MortonKey,
        point::{Point, PointType},
    },
};

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

#[no_mangle]
pub extern "C" fn distributed_tree_to_vtk(
    p_tree: *const DistributedTree,
    p_filename: *mut c_char,
    p_origin: *const [PointType; 3],
    p_diameter: *const [PointType; 3],
) {
    let origin = unsafe { std::slice::from_raw_parts(p_origin, 1) }[0];
    let diameter = unsafe { std::slice::from_raw_parts(p_diameter, 1) }[0];
    let filename = unsafe { CString::from_raw(p_filename).to_str().unwrap().to_string() };
    let tree = unsafe { &*p_tree };
    let domain = Domain { origin, diameter };
    tree.keys.write_vtk(filename, &domain);
}

#[no_mangle]
pub extern "C" fn distributed_tree_to_hdf5(
    p_tree: *const DistributedTree,
    p_filename: *mut c_char,
    balanced: bool,
) {
    let filename = unsafe { CString::from_raw(p_filename).to_str().unwrap().to_string() };
    let tree = unsafe { &*p_tree };

    &tree.keys;
}
