use mpi::{
    ffi::MPI_Comm,
    topology::UserCommunicator,
    traits::*
};

use crate::{
    types::domain::Domain,
    types::point::PointType
};


#[no_mangle]
pub extern "C" fn domain_from_local_points(p_points: *const [PointType; 3], len: usize) -> *mut Domain {
    let points = unsafe { std::slice::from_raw_parts(p_points, len) };
    let domain = Domain::from_local_points(points);

    Box::into_raw(Box::new(domain))
}

#[no_mangle]
pub extern "C" fn domain_from_global_points(p_points: *const [PointType; 3], len: usize, comm: MPI_Comm) -> *mut Domain {
    let points = unsafe { std::slice::from_raw_parts(p_points, len) };
    let comm = std::mem::ManuallyDrop::new(unsafe {UserCommunicator::from_raw(comm)}.unwrap());
    let domain = Domain::from_global_points(points, &comm);

    Box::into_raw(Box::new(domain))
}

