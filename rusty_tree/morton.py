"""Data Structures for Morton indices."""
import numpy as np
from rusty_tree import lib, ffi


class MortonKey(object):
    """
    Definition of a Morton Key
    """

    def __init__(self, p_key):
        """
        Initialize from a pointer to a Morton key.

        This constructor should not be used outside the class. Instead
        use the provided class methods to construct a MortonKey object.
        """

        self._p_key = p_key

    @property
    def ctype(self):
        """Give access to the underlying ctype."""

        return self._p_key

    @property
    def anchor(self):
        """Return the anchor."""
        return np.array([*self.ctype.anchor], dtype=np.uint64)

    @property
    def morton(self):
        """Return the Morton index."""
        return self.ctype.morton

    @classmethod
    def from_anchor(cls, anchor):
        """Create a Morton key from a given anchor."""
        anchor = np.array(anchor, dtype=np.uint64)
        data = ffi.from_buffer("uint64_t(*)[3]", anchor)
        return cls(lib.morton_key_from_anchor(data))

    @classmethod
    def from_morton(cls, morton):
        """Create a Morton key from a given Morton index."""
        return cls(lib.morton_key_from_morton(morton))

    @classmethod
    def from_point(cls, point, origin, diameter):
        """Create a Morton key from a point at the deepest level."""
        point = np.array(point, dtype=np.float64)
        point_data = ffi.from_buffer("double(*)[3]", point)

        origin = np.array(origin, dtype=np.float64)
        origin_data = ffi.from_buffer("double(*)[3]", origin)

        diameter = np.array(point, dtype=np.float64)
        diameter_data = ffi.from_buffer("double(*)[3]", diameter)

        return cls(lib.morton_key_from_point(point_data, origin_data, diameter_data))

    def __del__(self):
        """Destructor to ensure that the C memory is cleaned up."""
        lib.morton_key_delete(self.ctype)

    def parent(self):
        """Return the parent."""

        return MortonKey(lib.morton_key_parent(self.ctype))

    def first_child(self):
        """Return the first child."""
        return MortonKey(lib.morton_key_first_child(self.ctype))

    def children(self):
        """Return the children."""
        ptr = lib.morton_key_children(self.ctype)
        children = [MortonKey(ffi.cast('struct MortonKey *', ptr[index])) for index in range(8)]
        lib.delete_usize_vec(ptr, 8)
        return children

