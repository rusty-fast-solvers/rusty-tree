"""
Domains specify the simulation region from a globally distributed set of points.
"""
import numpy as np

from rusty_tree import lib, ffi


class Domain:
    def __init__(self, p_domain):
        """
        Initialize a domain from a pointer to a Domain struct in Rust.

        This constructor should not be used outside the class. Instead use the
        provided class methods to construct a Domain object.

        Parameters
        ----------
        p_domain: cdata 'struct <Domain> *'
        Pointer to a Domain struct initialized in Rust.
        """
        self._p_domain = p_domain

    @property
    def ctype(self):
        """Give access to the underlying ctype."""
        return self._p_domain

    @property
    def origin(self):
        """Coordinate corresponding to the origin of the domain."""
        return np.array([*self.ctype.origin], dtype=np.float64)

    @property
    def diameter(self):
        """Width of domain along each axis."""
        return np.array([*self.ctype.diameter], dtype=np.float64)

    @classmethod
    def from_local_points(cls, points):
        """Infer the domain from points on this processor."""
        points = np.array(points, dtype=np.float64, order="C")
        npoints, _ = points.shape
        points_data = ffi.from_buffer(f"double(*)[3]", points)
        return cls(lib.domain_from_local_points(points_data, npoints))

    @classmethod
    def from_global_points(cls, points, comm):
        """Infer the domain from points on all processors."""
        points = np.array(points, dtype=np.float64, order="C")
        npoints, _ = points.shape
        points_data = ffi.from_buffer(f"double(*)[3]", points)
        return cls(lib.domain_from_global_points(points_data, npoints, comm))
