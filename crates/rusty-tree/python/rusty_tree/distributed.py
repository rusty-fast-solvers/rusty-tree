import numpy as np
from rusty_tree import lib, ffi


class DistributedTree:

    def __init__(self, p_tree):
        self._p_tree = p_tree

    @classmethod
    def from_global_points(cls, points, balanced, comm):
        points = np.array(points, dtype=np.float64, order='C')
        npoints, _ = points.shape
        points_data = ffi.from_buffer(f"double(*)[][3]", points)
        balanced_data = ffi.cast('bool', balanced)
        npoints_data= ffi.cast('size_t', npoints)

        return cls(lib.distributed_tree_from_points(points_data, npoints_data, balanced_data, comm.raw))

    