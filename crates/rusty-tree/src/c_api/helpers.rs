use mpi::ffi::{MPI_Comm, MPI_Comm_free};

#[no_mangle]
pub extern "C" fn cleanup(comm: &mut MPI_Comm) 
{
    unsafe {MPI_Comm_free(comm)};
}