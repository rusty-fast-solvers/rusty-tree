
class Domain:
    
    def __init__(self, p_point):
        self._p_point = p_point

    @property
    def ctype(self):
        """Give access to the underlying ctype."""
        return self._p_point
    
    @property
    def origin(self):
        return np.array([*self.ctype.origin], dtype=np.float64)

    @property
    def diameter(self):
        return np.array([*self.ctype.diameter], dtype=np.float64)
