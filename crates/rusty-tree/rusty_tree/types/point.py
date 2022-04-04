import numpy as np

from rusty_tree import ffi, lib

class Point:

    def __init__(self, p_point):
        self._p_point = p_point

    def __eq__(self, other):
        """Implement == operator."""
        return self.key == other.key

    def __ne__(self, other):
        """Implement != operator."""
        return self.key != other.key

    def __lt__(self, other):
        """Implement < operator."""
        return self.key < other.key

    def __le__(self, other):
        """Implement <= operator."""
        return self.key <= other.key

    def __gt__(self, other):
        """Implement > operator."""
        return self.key > other.key

    def __ge__(self, other):
        """Implement >= operator."""
        return self.key >= other.key

    def __repr__(self):
        return str(
            {
                "coordinate": self.coordinate(), 
                "global_idx": self.global_idx(),
                "key": self.morton(),
                "anchor": self.anchor()
            }
        )

    @property
    def ctype(self):
        """Give access to the underlying ctype."""
        return self._p_point

    def coordinate(self):
        """Return the coordinate."""
        return np.array([*self.ctype.coordinate], dtype=np.float64)
    
    def global_idx(self):
        """Return the coordinate."""
        return self.ctype.global_idx

    def morton(self):
        """Return the associated Morton key"""
        return self.ctype.key.morton

    def anchor(self):
        return np.array([*self.ctype.key.anchor], dtype=np.uint64)


class Iterator():

    def __init__(self, pointer, n, type):
        self._pointer = pointer
        self._n = n
        self._head = pointer
        self._curr = self._head
        self._iter = 0
        self._n = n
        self.type = type
 
    @classmethod
    def from_points(cls, pointer, n):
        cls(pointer, n, Point)

    @classmethod
    def from_keys(cls, pointer, n):
        cls(pointer, n, MortonKey)
 
class Points:

    def __init__(self, pointer, n):
        self._pointer = pointer
        self._n = n
        self._head = pointer
        self._curr = self._head
        self._iter = 0
        self._n = n

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
        return 'Points '+str({'len': len(self), 'head': self.head})
    
    @property
    def curr(self):
        return Point(self._curr)

    @property
    def head(self):
        return Point(self._head)
    
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
        lib.point_slice(self._head, ptr_data, nkeys, start, stop)
        return [Point(ffi.cast('Point *', ptr[index])) for index in range(nslice)]
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, _ = key.indices(len(self))
            return self._slice(start, stop)    
        else:
            raise TypeError('Invalid argument type: {}'.format(type(key)))
