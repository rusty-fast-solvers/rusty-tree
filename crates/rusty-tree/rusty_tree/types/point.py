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
                "anchor": self.anchor(),
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
