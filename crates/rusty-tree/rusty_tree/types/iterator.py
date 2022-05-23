"""
Python Iterators that help to manipulate Rust Iterators without copying data
into Numpy arrays.
"""
import numpy as np

from rusty_tree import lib, ffi
from rusty_tree.types.morton import MortonKey
from rusty_tree.types.point import Point


class IteratorProtocol:
    """
    Wrapper defining an Iterator protocol implemented via Rust functions
    exposed via the CFFI.
    """

    def __init__(self, p_type, c_name, clone, next, index):
        """
        Params:
        -------
        p_type : Object
            Python object that mirrors a struct from Rust.
        c_name : str
            The name of the Rust struct.
        clone : _cffi_backend._CDataBase
            CFFI function for cloning a slice of a Rust iterator.
        next : _cffi_backend._CDataBase
            CFFI function for advancing a single element in a Rust iterator.
        index : _cffi_backend._CDataBase
            CFFI function for indexing a single element from a Rust iterator.
        """
        self.p_type = p_type
        self.c_name = c_name
        self.clone = clone
        self.next = next
        self.index = index


MortonProtocol = IteratorProtocol(
    p_type=MortonKey,
    c_name="MortonKey",
    clone=lib.morton_key_clone,
    next=lib.morton_key_next,
    index=lib.morton_key_index,
)

PointProtocol = IteratorProtocol(
    p_type=Point,
    c_name="Point",
    clone=lib.point_clone,
    next=lib.point_next,
    index=lib.point_index,
)


class Iterator:
    """
    Wrapper for Rust iterators exposed via a raw pointer via CFFI.
    """

    def __init__(self, pointer, n, iterator_protocol):
        """
        This constructor should not be used outside the class. Instead
        use the provided class methods to construct an Iterator object.

        Parameters
        ----------
        pointer : cdata 'struct <Vec<T>> *'
            Pointer to a the first element in a Vec<T> in Rust where type 'T'
            has been exposed in Python.
        n : int
            Number of elements in Vec<T>.
        iterator_protocol : IteratorProtocol
            Helper class exposing the Rust iterator in Python.
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
        """
        Construct an Iterator for an exposed Vec<Point>.

        Parameters
        ----------
        pointer : cdata 'struct <T> *'
            Pointer to a the first element in a Vec<T> in Rust where type 'T'
            has been exposed in Python.
        n : int
            Number of elements in Vec<T>.

        Returns
        -------
        Iterator
            Instance of Rust iterator now wrapped in Python.
        """
        return cls(pointer, n, PointProtocol)

    @classmethod
    def from_keys(cls, pointer, n):
        """
        Construct an Iterator for an exposed Vec<MortonKey>.

        Parameters
        ----------
        pointer : cdata 'struct <T> *'
            Pointer to a the first element in a Vec<T> in Rust where type 'T'
            has been exposed in Python.
        n : int
            Number of elements in Vec<T>.

        Returns
        -------
        Iterator
            Instance of Rust iterator now wrapped in Python.
        """
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
        """Printing to stdout forces a copy."""
        return str(self._clone(0, len(self)))

    @property
    def head(self):
        """Return head of iterator, wrapped in a new compatible Python type."""
        return self._iterator_protocol.p_type(self._head)

    @property
    def ctype(self):
        """Return the current head"""
        return self._head

    def _index(self, index):
        """
        Index into an element of the exposed Vec<T> without copy.
        """
        index = ffi.cast("size_t", index)
        ntot = ffi.cast("size_t", len(self))
        return self._iterator_protocol.index(self._head, ntot, index)

    def _clone(self, start, stop):
        """Clone a slice of the exposed Vec<T> into a Python datatype."""
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
        """
        Index into a slice of the exposed Vec<T> without copy by returning the
        slice inside a new Python Iterator.
        """
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
