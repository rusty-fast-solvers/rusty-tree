"""
Distributed Tree.
"""
from mpi4py import MPI
import numpy as np

from rusty_tree import lib, ffi
from rusty_tree.types.morton import MortonKey
from rusty_tree.types.point import Point
from rusty_tree.types.iterator import Iterator


class DistributedTree:
    """
    Wrapper a DistributedTree structure from Rust.

    Example Usage:
    --------------
    >>> from mpi4py import MPI
    >>> import numpy as np

    >>> from rusty_tree.distributed import DistributedTree

    >>> # Setup MPI communicator
    >>> comm = MPI.COMM_WORLD

    >>> # Initialize points at the current processor
    >>> points = np.random.rand(1000, 3)

    >>> # Create a balanced, distributed, tree from a set of globally distributed points
    >>> tree = DistributedTree.from_global_points(points, True, comm)
    """

    def __init__(self, p_tree, comm, p_comm, raw_comm):
        """
        Don't directly use constructor, instead use the provided class method to create
        a DistributedTree from a set of points, distributed globally across the set of
        processors provided to the constructor via its communicator.
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

    @classmethod
    def from_global_points(cls, points, balanced, comm):
        """
        Construct a distributed tree from a set of globally distributed points.

        Params:
        ------
        points : np.array(shape=(n_points, 3), dtype=np.float64)
            Cartesian points at this processor.
        balanced : bool
            If 'True' constructs a balanced tree, if 'False' constructs an unbalanced tree.
        comm : Intracomm
           An mpi4py Intracommunicator.
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