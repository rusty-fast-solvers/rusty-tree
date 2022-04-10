import numpy as np

from rusty_tree import lib, ffi


class MortonKey:
    def __init__(self, p_key):
        """
        Initialize a Morton key from a pointer to a MortonKey struct in Rust.

        This constructor should not be used outside the class. Instead
        use the provided class methods to construct a MortonKey object.
        """
        self._p_key = p_key

    def __del__(self):
        """Destructor to ensure that the C memory is cleaned up."""
        lib.morton_key_delete(self.ctype)

    def __repr__(self):
        return str({"morton": self.morton(), "anchor": self.anchor()})

    def __eq__(self, other):
        """Implement == operator."""
        return self.morton == other.morton

    def __ne__(self, other):
        """Implement != operator."""
        return self.morton != other.morton

    def __lt__(self, other):
        """Implement < operator."""
        return self.morton < other.morton

    def __le__(self, other):
        """Implement <= operator."""
        return self.morton <= other.morton

    def __gt__(self, other):
        """Implement > operator."""
        return self.morton > other.morton

    def __ge__(self, other):
        """Implement >= operator."""
        return self.morton >= other.morton

    def __hash__(self):
        return self.morton()

    @property
    def ctype(self):
        """Give access to the underlying ctype."""
        return self._p_key

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

        diameter = np.array(diameter, dtype=np.float64)
        diameter_data = ffi.from_buffer("double(*)[3]", diameter)

        return cls(lib.morton_key_from_point(point_data, origin_data, diameter_data))

    def anchor(self):
        """Return the anchor."""
        return np.array([*self.ctype.anchor], dtype=np.uint64)

    def morton(self):
        """Return the Morton index."""
        return self.ctype.morton

    def level(self):
        """Return the level."""
        return lib.morton_key_level(self.ctype)

    def parent(self):
        """Return the parent."""
        return MortonKey(lib.morton_key_parent(self.ctype))

    def first_child(self):
        """Return the first child."""
        return MortonKey(lib.morton_key_first_child(self.ctype))

    def children(self):
        """Return the children."""
        ptr = np.empty(8, dtype=np.uint64)
        ptr_data = ffi.from_buffer("uintptr_t *", ptr)
        lib.morton_key_children(self.ctype, ptr_data)
        children = [
            MortonKey(ffi.cast("MortonKey *", ptr[index])) for index in range(8)
        ]
        return children

    def ancestors(self):
        curr = self
        ancestors = set()
        while curr.morton() != 0:
            parent = curr.parent()
            ancestors.add(parent)
            curr = parent
        return ancestors

    def siblings(self):
        """Return all children of the parent."""
        return self.parent().children()

    def is_ancestor(self, other):
        """Check if the key is ancestor of `other`."""
        return lib.morton_key_is_ancestor(self.ctype, other.ctype)

    def is_descendent(self, other):
        """Check if the key is descendent of `other`."""
        return lib.morton_key_is_descendent(self.ctype, other.ctype)

    def to_coordinates(self, origin, diameter):
        """Return the coordinates of the anchor."""
        coords = np.empty(3, dtype=np.float64)
        coords_data = ffi.from_buffer("double(*)[3]", coords)
        origin = np.array(origin, dtype=np.float64)
        origin_data = ffi.from_buffer("double(*)[3]", origin)
        diameter = np.array(diameter, dtype=np.float64)
        diameter_data = ffi.from_buffer("double(*)[3]", diameter)

        lib.morton_key_to_coordinates(
            self.ctype, origin_data, diameter_data, coords_data
        )

        return coords

    def box_coordinates(self, origin, diameter):
        """
        Return the 8 coordinates of the box associated with the key.

        Let the unit cube be described as follows:

        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
         ]

        The function returns the box coordinates in the same order
        as the above unit cube coordinates.

        """
        coords = np.empty(24, dtype=np.float64)
        coords_data = ffi.from_buffer("double(*)[24]", coords)

        origin = np.array(origin, dtype=np.float64)
        origin_data = ffi.from_buffer("double(*)[3]", origin)
        diameter = np.array(diameter, dtype=np.float64)
        diameter_data = ffi.from_buffer("double(*)[3]", diameter)

        lib.morton_key_box_coordinates(
            self.ctype, origin_data, diameter_data, coords_data
        )

        return coords.reshape(8, 3)

    def find_key_in_direction(self, direction):
        """
        Find a key in a given direction.

        Given an integer list `direction` containing
        3 elements, return the key obtained by moving
        from the current key `direction[j]` steps along
        dimension [j]. For example, if `direction = [2, -1, 1]`
        the method returns the key by moving two boxes in positive
        x-direction, one box in the negative y direction and one box
        in the positive z direction. Boxes are counted with respect to
        the current level.

        If there is no box in the given direction, i.e. the new coordinates
        are out of bounds, the method retunrs None.

        """

        direction = np.array(direction, dtype=np.int64)
        direction_data = ffi.from_buffer("int64_t(*)[3]", direction)

        ptr = lib.morton_key_key_in_direction(self.ctype, direction_data)

        if ptr == ffi.NULL:
            return None
        else:
            return MortonKey(ptr)
