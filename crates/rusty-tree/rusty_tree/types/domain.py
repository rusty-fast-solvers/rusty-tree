"""Data Structure for Morton Key."""
import numpy as np
from rusty_tree import lib, ffi


class Domain:
    def __init__(self, p_domain):
        """
        Initialize a domain from an origin and a diameter.

        This constructor should not be used outside the class. Instead
        use the provided class methods to construct a Domain object.
        """
        self._p_domain = p_domain

    @property
    def ctype(self):
        """Give access to the underlying ctype."""
        return self._p_domain

    @property
    def origin(self):
        return np.array([*self.ctype.origin], dtype=np.float64)

    @property
    def diameter(self):
        return np.array([*self.ctype.diameter], dtype=np.float64)

    @classmethod
    def from_local_points(cls, points):
        points = np.array(points, dtype=np.float64, order="C")
        npoints, _ = points.shape
        points_data = ffi.from_buffer(f"double(*)[3]", points)
        n_points_data = ffi.cast("size_t", npoints)
        return cls(lib.domain_from_local_points(points_data, npoints))

    @classmethod
    def from_global_points(cls, points, comm):
        points = np.array(points, dtype=np.float64, order="C")
        npoints, _ = points.shape
        points_data = ffi.from_buffer(f"double(*)[3]", points)
        n_points_data = ffi.cast("size_t", npoints)
        return cls(lib.domain_from_global_points(points_data, npoints, comm))
