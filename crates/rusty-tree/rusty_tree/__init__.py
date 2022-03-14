"""
rusty-tree: Large-scale parallel octrees
"""

# Deepest Level
DEEPEST_LEVEL = 16

# The level displacement in a Morton index in bits
LEVEL_DISPLACEMENT = 15

from .rusty_tree import ffi, lib


