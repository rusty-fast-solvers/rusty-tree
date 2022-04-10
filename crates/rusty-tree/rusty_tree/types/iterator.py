import numpy as np

from rusty_tree import lib, ffi
from rusty_tree.types.morton import MortonKey
from rusty_tree.types.point import Point


class IteratorProtocol:
    """
    Wrapper defining an Iterator protocol implemented via Rust
    functions exposed via the CFFI.
    """

    def __init__(self, p_type, c_name, clone, next, index):
        self.p_type = p_type
        self.c_name = c_name
        self.clone = clone
        self.next = next
        self.index = index


MortonProtocol = IteratorProtocol(
    MortonKey,
    "MortonKey",
    lib.morton_key_clone,
    lib.morton_key_next,
    lib.morton_key_index,
)

PointProtocol = IteratorProtocol(
    Point, "Point", lib.point_clone, lib.point_next, lib.point_index
)


class Iterator:
    """
    Wrapper for Rust iterators exposed via a raw pointer via CFFI.
    """

    def __init__(self, pointer, n, iterator_protocol):
        """
        This constructor should not be used outside the class. Instead
        use the provided class methods to construct an Iterator object.
        """
        self._pointer = pointer
        self._n = n
        self._head = pointer
        self._curr = self._head
        self._iter = 0
        self._n = n
        self._iterator_protocol = iterator_protocol

    @classmethod
    def from_points(cls, pointer, n):
        """Construct an Iterator for an exposed list of Points"""
        return cls(pointer, n, PointProtocol)

    @classmethod
    def from_keys(cls, pointer, n):
        """Construct an Iterator for an exposed list of Keys"""
        return cls(pointer, n, MortonProtocol)

    def __len__(self):
        return self._n

    def __iter__(self):
        self._curr = self._head
        self._iter = 0
        return self

    def __next__(self):
        _curr = self._curr
        _next = self._iterator_protocol.next(self._curr)[0]

        if self._iter < len(self):
            if _curr != _next:
                self._curr = _next
                self._iter += 1
                return _curr
        else:
            raise StopIteration

    def __repr__(self):
        return str(self._clone(0, len(self)))

    @property
    def head(self):
        return self._iterator_protocol.p_type(self._head)

    @property
    def ctype(self):
        return self._head

    def _index(self, index):
        index = ffi.cast("size_t", index)
        ntot = ffi.cast("size_t", len(self))
        return self._iterator_protocol.index(self._head, ntot, index)

    def _clone(self, start, stop):
        """Clone a slice into a Python datatype"""
        n = ffi.cast("size_t", len(self))
        nslice = stop - start
        start = ffi.cast("size_t", start)
        stop = ffi.cast("size_t", stop)
        ptr = np.empty(nslice, dtype=np.uint64)
        ptr_data = ffi.from_buffer("uintptr_t *", ptr)
        self._iterator_protocol.clone(self._head, ptr_data, n, start, stop)
        return [
            self._iterator_protocol.p_type(
                ffi.cast(f"{self._iterator_protocol.c_name} *", ptr[index])
            )
            for index in range(nslice)
        ]

    def _slice(self, start, stop):
        nslice = stop - start
        ptr = self._index(start)[0]
        return Iterator(ptr, nslice, self._iterator_protocol)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, _ = key.indices(len(self))
            return self._slice(start, stop)

        elif isinstance(key, int):
            ptr = self._slice(key, key + 1)
            return ptr

        else:
            raise TypeError("Invalid argument type: {}".format(type(key)))
