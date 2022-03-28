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
        point::PointType
    }
};

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
    let domain = Domain::from_global_points(points, &comm);

    // let universe = Universe::world();
    Box::into_raw(Box::new(DistributedTree::new(points, &domain, balanced, &mut *comm)))
}