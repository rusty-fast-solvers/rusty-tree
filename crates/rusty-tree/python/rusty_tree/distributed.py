import numpy as np
from rusty_tree import lib, ffi
from rusty_tree.types.morton import MortonKey


class DistributedTree:

    def __init__(self, p_tree):
        self._p_tree = p_tree
 
    @property
    def ctype(self):
        """Give access to the underlying ctype."""
        return self._p_tree

    @classmethod
    def from_global_points(cls, points, balanced, comm):
        points = np.array(points, dtype=np.float64, order='C')
        npoints, _ = points.shape
        points_data = ffi.from_buffer(f"double(*)[][3]", points)
        balanced_data = ffi.cast('bool', np.bool(balanced))
        npoints_data = ffi.cast('size_t', npoints)

        return cls(lib.distributed_tree_from_points(points_data, npoints_data, balanced_data, comm.raw))

    @property
    def nkeys(self):
        nkeys = lib.distributed_tree_n_keys(self.ctype)
        return nkeys

    @property
    def keys(self):
        ptr = np.empty(self.nkeys, dtype=np.uint64)
        ptr_data = ffi.from_buffer('uintptr_t *', ptr)
        lib.distributed_tree_keys(self.ctype, ptr_data)
        keys = [MortonKey(ffi.cast('MortonKey *', ptr[index])) for index in range(self.nkeys)]
        return keys