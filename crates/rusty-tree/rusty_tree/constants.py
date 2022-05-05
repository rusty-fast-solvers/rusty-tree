"""
Project wide constants.
"""
import os

import numpy as np

from rusty_tree import lib, ffi
from rusty_tree.types.morton import MortonKey


def load_cbuffer(c_buffer, shape, dtype):
    """Load C Buffer into a Numpy Buffer"""
    size = 8
    for dim in shape:
        size = dim * size

    return np.frombuffer(ffi.buffer(c_buffer, size), dtype).reshape(shape)


DIRECTIONS = load_cbuffer(lib.DIRECTIONS, (26, 3), np.int64)
Z_LOOKUP_ENCODE = load_cbuffer(lib.Z_LOOKUP_ENCODE, (256,), np.uint64)
Z_LOOKUP_DECODE = load_cbuffer(lib.Z_LOOKUP_DECODE, (512,), np.uint64)
Y_LOOKUP_ENCODE = load_cbuffer(lib.Y_LOOKUP_ENCODE, (256,), np.uint64)
Y_LOOKUP_DECODE = load_cbuffer(lib.Y_LOOKUP_DECODE, (512,), np.uint64)
X_LOOKUP_ENCODE = load_cbuffer(lib.X_LOOKUP_ENCODE, (256,), np.uint64)
X_LOOKUP_DECODE = load_cbuffer(lib.X_LOOKUP_DECODE, (512,), np.uint64)


DEEPEST_LEVEL = 16
ROOT = MortonKey.from_morton(np.uint64(0))
LEVEL_SIZE = 1 << DEEPEST_LEVEL

# Parse from Environment
K = os.getenv("K", 2)
NCRIT = os.getenv("NCRIT", 150)
