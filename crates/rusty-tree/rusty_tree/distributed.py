import numpy as np
from rusty_tree import lib, ffi
from rusty_tree.types.morton import MortonKey
from rusty_tree.types.point import Point


class DistributedTree:

    def __init__(self, p_tree):
        self._p_tree = p_tree
 
    @property
    def ctype(self):
        """Give access to the underlying ctype."""
        return self._p_tree

    @classmethod
    def from_global_points(cls, points, balanced, comm):
        points = np.array(points, dtype=np.float64, order='C', copy=False)
        npoints, _ = points.shape
        points_data = ffi.from_buffer(f"double(*)[3]", points)
        balanced_data = ffi.cast('bool', np.bool(balanced))
        npoints_data = ffi.cast('size_t', npoints)

        return cls(lib.distributed_tree_from_points(points_data, npoints_data, balanced_data, comm))

    @classmethod
    def random(cls, balanced, npoints, comm):
        balanced_data = ffi.cast('bool', np.bool(balanced))
        npoints_data = ffi.cast('size_t', npoints)
        return cls(lib.distributed_tree_random(balanced_data, npoints_data, comm))

    @property
    def nkeys(self):
        return lib.distributed_tree_n_keys(self.ctype)

    def keys_slice(self, lidx, ridx):
        lidx_data = ffi.cast('size_t', lidx)
        ridx_data = ffi.cast('size_t', ridx)
        nkeys = ridx-lidx
        ptr = np.empty(nkeys, dtype=np.uint64)
        ptr_data = ffi.from_buffer('uintptr_t *', ptr)
        lib.distributed_tree_keys_slice(self.ctype, ptr_data, lidx_data, ridx_data)
        slice = [MortonKey(ffi.cast('MortonKey *', ptr[index])) for index in range(nkeys)]
        return slice
    
    def points_slice(self, lidx, ridx):
        lidx_data = ffi.cast('size_t', lidx)
        ridx_data = ffi.cast('size_t', ridx)
        npoints = ridx-lidx
        ptr = np.empty(npoints, dtype=np.uint64)
        ptr_data = ffi.from_buffer('uintptr_t *', ptr)
        lib.distributed_tree_points_slice(self.ctype, ptr_data, lidx_data, ridx_data)
        slice = [Point(ffi.cast('Point *', ptr[index])) for index in range(npoints)]
        return slice

    @property
    def balanced(self):
        return lib.distributed_tree_balanced(self.ctype)
    
    @property
    def npoints(self):
        return lib.distributed_tree_n_points(self.ctype)
