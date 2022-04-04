import numpy as np

from rusty_tree import lib, ffi
from rusty_tree.types.morton import MortonKey
from rusty_tree.types.point import Point

C_TYPE_HEADERS = {
    MortonKey: 'MortonKey',
    Point: 'Point'
}

SLICE_FUNCS = {
    MortonKey: lib.morton_key_slice,
    Point: lib.point_slice
}

class Iterator:
    def __init__(self, pointer, n, type):
        self._pointer = pointer
        self._n = n
        self._head = pointer
        self._curr = self._head
        self._iter = 0
        self._n = n
        self.type = type
        self.ctype_str = C_TYPE_HEADERS[type]
        self.slice_func = SLICE_FUNCS[type]
 
    @classmethod
    def from_points(cls, pointer, n):
        cls(pointer, n, Point)

    @classmethod
    def from_keys(cls, pointer, n):
        cls(pointer, n, MortonKey)
 
    def __len__(self):
        return self._n

    def __iter__(self):
        self._curr = self._head
        self._iter = 0
        return self

    def __next__(self):
        _curr = self._curr
        _next = lib.point_next(self._curr)[0]

        if self._iter < len(self):
            if _curr != _next:
                self._curr = _next
                self._iter += 1
                return _curr
        else:
            raise StopIteration
    
    def __repr__(self):
        return f'{self.ctype_str} '+str({'len': len(self), 'head': self.head})
    
    @property
    def curr(self):
        return self.type(self._curr)

    @property
    def head(self):
        return self.type(self._head)
    
    @property
    def ctype(self):
        return self._head

    def _slice(self, start, stop):
        nslice = stop-start
        nkeys = ffi.cast('size_t', len(self))
        start = ffi.cast('size_t', start)
        stop = ffi.cast('size_t', stop)
        ptr = np.empty(nslice, dtype=np.uint64)
        ptr_data = ffi.from_buffer('uintptr_t *', ptr)
        self.slice_func(self._head, ptr_data, nkeys, start, stop)
        return [self.type(ffi.cast(f'{self.ctype_str} *', ptr[index])) for index in range(nslice)]
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, _ = key.indices(len(self))
            return self._slice(start, stop)    
        else:
            raise TypeError('Invalid argument type: {}'.format(type(key)))
