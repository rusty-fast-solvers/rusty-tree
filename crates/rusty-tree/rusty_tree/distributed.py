"""
The DistributedTree object acts as the interface for creating octrees.
"""
from mpi4py import MPI
import numpy as np

from rusty_tree import lib, ffi
from rusty_tree.types.iterator import Iterator
from rusty_tree.types.morton import MortonKey
from rusty_tree.types.point import Point


class DistributedTree:
    """
    Wrapper for a DistributedTree structure in Rust. Used to create octrees
    distributed via MPI from a set of N distributed Cartesian points with shape
    (N,3) stored in NumPy arrays on each processor. Trees can optionally be
    balanced.
    """

    def __init__(self, p_tree, comm, p_comm, raw_comm):
        """
        Don't directly use constructor, instead use the provided class methods
        to create a DistributedTree from a set of points, distributed globally
        across the set of processors provided to the constructor via its
        communicator.

        Parameters
        ----------
        p_tree: cdata 'struct <DistributedTree> *'
            Pointer to a DistributedTree struct initialized in Rust.
        comm: mpi4py.MPI.Intracomm
            MPI world communicator, created using mpi4py.
        p_comm: cdata '*MPI_Comm'
            Pointer to underlying C communicator.
        raw_comm: cdata 'MPI_Comm'
            Underlying C communicator.
        """
        self.ctype = p_tree
        self.nkeys = lib.distributed_tree_nkeys(self.ctype)
        self.keys = Iterator.from_keys(
            lib.distributed_tree_keys(self.ctype), self.nkeys
        )
        self.npoints = lib.distributed_tree_npoints(self.ctype)
        self.points = Iterator.from_points(
            lib.distributed_tree_points(self.ctype), self.npoints
        )
        self.balanced = lib.distributed_tree_balanced(self.ctype)
        self.comm = comm
        self.p_comm = p_comm
        self.raw_comm = raw_comm

    def __getitem__(self, key):
        if isinstance(key, Point):
            return MortonKey(lib.distributed_tree_points_to_keys_get(self.ctype, key.ctype))
        
        elif isinstance(key, MortonKey):
            n_points = lib.distributed_tree_keys_to_npoints_get(self.ctype, key.ctype)
            ptr = lib.distributed_tree_keys_to_points_get(self.ctype, key.ctype)
            return Iterator.from_points(ptr, n_points)

        else:
            raise TypeError(
                "Invalid key type {}, only Point or MortonKey accepted".format(type(key))
                )

    @classmethod
    def from_global_points(cls, points, balanced, comm):
        """
        Construct a distributed tree from a set of globally distributed points.

        Parameters
        ----------
        points : np.array(shape=(n_points, 3), dtype=np.float64)
            Cartesian points at this processor.
        balanced : bool
            If 'True' constructs a balanced tree, if 'False' constructs an unbalanced tree.
        comm: mpi4py.MPI.Intracomm
            MPI world communicator, created using mpi4py.

        Returns
        -------
        DistributedTree

        Example Usage:
        --------------
        >>> from mpi4py import MPI
        >>> import numpy as np

        >>> from rusty_tree.distributed import DistributedTree

        >>> # Setup MPI communicator
        >>> comm = MPI.COMM_WORLD

        >>> # Initialize points at the current processor
        >>> points = np.random.rand(1000, 3)

        >>> # Create a balanced, distributed, tree from a set of globally
        >>> # distributed points
        >>> tree = DistributedTree.from_global_points(points, True, comm)
        """
        points = np.array(points, dtype=np.float64, order="C", copy=False)
        npoints, _ = points.shape
        points_data = ffi.from_buffer(f"double(*)[3]", points)
        balanced_data = ffi.cast("bool", np.bool(balanced))
        npoints_data = ffi.cast("size_t", npoints)
        p_comm = MPI._addressof(comm)
        raw_comm = ffi.cast("uintptr_t*", p_comm)

        return cls(
            lib.distributed_tree_from_points(
                points_data, npoints_data, balanced_data, raw_comm
            ),
            comm,
            p_comm,
            raw_comm,
        )

    @classmethod
    def read_hdf5(cls, filepath, comm):
        """
        Instantiate a tree from tree data serialized with HDF5 on the master node,
        and distribute over processes in provided communicator.

        Parameters
        ----------
        filepath: str

        Returns
        -------
        DistributedTree

        Example Usage:
        --------------
        >>> from mpi4py import MPI
        >>> import numpy as np

        >>> from rusty_tree.distributed import DistributedTree

        >>> # Setup MPI communicator
        >>> comm = MPI.COMM_WORLD

        >>> # Read a balanced, distributed tree from disk
        >>> tree = DistributedTree.read_hdf5('/path/to/file.hdf5', comm)
        """

        filepath_data = ffi.new("char[]", filepath.encode("ascii"))
        p_filepath = ffi.cast("char *", ffi.addressof(filepath_data))
        p_comm = MPI._addressof(comm)
        raw_comm = ffi.cast("uintptr_t*", p_comm)

        return cls(
            lib.distributed_tree_read_hdf5(raw_comm, p_filepath), comm, p_comm, raw_comm
        )

    def write_vtk(self, filename):
        """
        Serialize leaves of a distributed tree in VTK format for visualization
        on the master node.

        Parameters
        ----------
        filename: str
        """
        filename_data = ffi.new("char[]", filename.encode("ascii"))
        p_filename = ffi.cast("char *", ffi.addressof(filename_data))

        lib.distributed_tree_write_vtk(self.raw_comm, self.ctype, p_filename)

    def write_hdf5(self, filename):
        """
        Serialize a distributed tree in HDF5 format on the master node.

        Parameters
        ----------
        filename: str
        """
        filename_data = ffi.new("char[]", filename.encode("ascii"))
        p_filename = ffi.cast("char *", ffi.addressof(filename_data))

        lib.distributed_tree_write_hdf5(self.raw_comm, self.ctype, p_filename)
