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

use std::ptr;

// pub struct Universe {
//     pub buffer: Vec<u8>
// }

// impl Universe {
//     fn world(&self) -> SystemCommunicator {
//         SystemCommunicator::world()
//     }
// }


#[no_mangle]
pub extern "C" fn distributed_tree_from_points(
    p_points: *const [PointType; 3],
    npoints: usize,
    balanced: bool,
    comm: MPI_Comm
) -> *mut DistributedTree {
    let points = unsafe { std::slice::from_raw_parts(p_points, npoints) }; 
    let mut comm = std::mem::ManuallyDrop::new(unsafe {UserCommunicator::from_raw(comm)}.unwrap());
    println!("HERE {:?}", balanced);
    // let universe = Universe::world();
    Box::into_raw(Box::new(DistributedTree::new(points, balanced, &mut comm)))
}

#[no_mangle]
pub extern "C" fn distributed_tree_n_keys(
    p_tree: *const DistributedTree,
) -> usize {
    let mut tree = unsafe { &*p_tree };
    tree.keys.len()
}


#[no_mangle]
pub extern "C" fn distributed_tree_keys(
    p_tree: *const DistributedTree,
    ptr: *mut usize
) {
    let mut tree = unsafe { &*p_tree };
    let mut keys = &tree.keys;
    let nkeys = keys.len();

    let boxes = unsafe {std::slice::from_raw_parts_mut(ptr, nkeys)};

    for index in 0..nkeys {
        let key = keys[index].clone();
        boxes[index] = Box::into_raw(Box::new(key)) as usize;
    }
}