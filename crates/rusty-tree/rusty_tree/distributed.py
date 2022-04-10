"""
Distributed Tree.
"""
import numpy as np

from rusty_tree import lib, ffi
from rusty_tree.types.morton import MortonKey
from rusty_tree.types.point import Point
from rusty_tree.types.iterator import Iterator


class DistributedTree:
    """
    Wrap a DistributedTree structure from Rust. Don't directly use constructor,
    instead use the provided class method to create a DistributedTree from a
    set of points, distributed globally across the set of processors provided
    to the constructor via its communicator.
    """

    def __init__(self, p_tree):
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

    @classmethod
    def from_global_points(cls, points, balanced, comm):
        """
        Construct a distributed tree from a set of globally distributed points
        """
        points = np.array(points, dtype=np.float64, order="C", copy=False)
        npoints, _ = points.shape
        points_data = ffi.from_buffer(f"double(*)[3]", points)
        balanced_data = ffi.cast("bool", np.bool(balanced))
        npoints_data = ffi.cast("size_t", npoints)
        return cls(
            lib.distributed_tree_from_points(
                points_data, npoints_data, balanced_data, comm
            )
        )
