"""
Points are associated with a Morton Key denoting the Octree node that they
belong to, which also defines their ordering. Additionally, Points are
associated with a unique 'global index', for ease of reference.
"""
import numpy as np


class Point:
    def __init__(self, p_point):
        """
        Initialize a Point from a pointer to a Point struct in Rust.

        Parameters
        ----------
        p_point: cdata 'struct <Point> *'
            Pointer to a Point struct initialized in Rust.
        """
        self._p_point = p_point

    def __eq__(self, other):
        return self.key == other.key

    def __ne__(self, other):
        return self.key != other.key

    def __lt__(self, other):
        return self.key < other.key

    def __le__(self, other):
        return self.key <= other.key

    def __gt__(self, other):
        return self.key > other.key

    def __ge__(self, other):
        return self.key >= other.key

    def __repr__(self):
        return str(
            {
                "coordinate": self.coordinate(),
                "global_idx": self.global_idx(),
                "key": self.morton(),
                "anchor": self.anchor(),
            }
        )

    @property
    def ctype(self):
        """Give access to the underlying ctype."""
        return self._p_point

    def coordinate(self):
        """Return the coordinate, with copy."""
        return np.array([*self.ctype.coordinate], dtype=np.float64)

    def global_idx(self):
        """Return the coordinate, without copy."""
        return self.ctype.global_idx

    def morton(self):
        """Return the associated Morton Key, without copy."""
        return self.ctype.key.morton

    def anchor(self):
        """Return the associated anchor, with copy."""
        return np.array([*self.ctype.key.anchor], dtype=np.uint64)
