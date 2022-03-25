from rusty_tree.morton import MortonKey


class Point:

    def __init__(self, p_point):
        self._p_point = p_point

    @property
    def ctype(self):
        """Give access to the underlying ctype."""
        return self._p_point

    @property
    def coordinate(self):
        """Return the coordinate."""
        return np.array([*self.ctype.coordinate], dtype=np.float64)
    
    @property
    def global_idx(self):
        """Return the coordinate."""
        return self.ctype.global_idx

    @property
    def key(self):
        """Return the associated Morton key"""
        return MortonKey.from_morton(self.ctype.morton)

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
