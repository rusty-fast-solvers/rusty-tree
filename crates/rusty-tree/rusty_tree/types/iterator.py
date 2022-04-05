import numpy as np

from rusty_tree import lib, ffi
from rusty_tree.types.morton import MortonKey
from rusty_tree.types.point import Point


class IteratorProtocol:
    """
    Wrapper for a Rust Iterator protocol.
    """
    def __init__(self, type, c_name, slice, next, index):
        self.p_type = type
        self.c_name = c_name
        self.slice = slice
        self.next = next
        self.index = index

MortonProtocol = IteratorProtocol(
    MortonKey, 
    'MortonKey', 
    lib.morton_key_slice, 
    lib.morton_key_next, 
    lib.morton_key_index
)

PointProtocol = IteratorProtocol(
    Point, 
    'Point', 
    lib.point_slice, 
    lib.point_next, 
    lib.point_index
)

class Iterator:
    """
    Wrapper for Rust iterators exposed via a raw pointer via CFFI.
    """
    def __init__(self, pointer, n, iterator_protocol):
        self._pointer = pointer
        self._n = n
        self._head = pointer
        self._curr = self._head
        self._iter = 0
        self._n = n
        self.iterator_protocol = iterator_protocol

    @classmethod
    def from_points(cls, pointer, n):
        return cls(pointer, n, PointProtocol)

    @classmethod
    def from_keys(cls, pointer, n):
        return cls(pointer, n, MortonProtocol)
 
    def __len__(self):
        return self._n

    def __iter__(self):
        self._curr = self._head
        self._iter = 0
        return self

    def __next__(self):
        _curr = self._curr
        _next = self.iterator_protocol.next(self._curr)[0]

        if self._iter < len(self):
            if _curr != _next:
                self._curr = _next
                self._iter += 1
                return _curr
        else:
            raise StopIteration
    
    def __repr__(self):
        nslice = len(self)
        n = ffi.cast('size_t', len(self))
        start = ffi.cast('size_t', 0)
        stop = ffi.cast('size_t', n)
        ptr = np.empty(nslice, dtype=np.uint64)
        ptr_data = ffi.from_buffer('uintptr_t *', ptr)
        self.iterator_protocol.slice(self._head, ptr_data, n, start, stop)
        return str([
            self.iterator_protocol.p_type(
           ffi.cast(f'{self.iterator_protocol.c_name} *', ptr[index])
           )
           for index in range(nslice)
           ])

    @property
    def head(self):
        return self.iterator_protocol.p_type(self._head)
    
    @property
    def ctype(self):
        return self._head

    def _index(self, index):
        index = ffi.cast('size_t', index)
        ntot = ffi.cast('size_t', len(self))
        return self.iterator_protocol.index(self._head, ntot, index)

    def _slice(self, start, stop):
        nslice = stop-start
        ptr = self._index(start)[0]
        return Iterator(ptr, nslice, self.iterator_protocol) 
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, _ = key.indices(len(self))
            return self._slice(start, stop)    

        elif isinstance(key, int):
            ptr = self._slice(key, key+1)
            return ptr

        else:
            raise TypeError('Invalid argument type: {}'.format(type(key)))
