"""
Large scale distributed octrees.
"""
import os
import platform

from cffi import FFI
from mpi4py import MPI


HERE = os.path.abspath(os.path.dirname(__file__))
LIBDIR = os.path.join(HERE, 'lib')

sysname = platform.system()

if sysname == 'Darwin':
    lib_name = 'librusty_tree.dylib'
elif sysname == 'Windows':
    lib_name = 'librusty_tree.dll'
else:
    lib_name = 'librusty_tree.so'


ffi = FFI()

types = """
typedef uint64_t KeyType;
typedef double PointType;

typedef struct {
    KeyType anchor[3];
    KeyType morton;
} MortonKey;
"""

methods = """
void morton_key_delete(MortonKey *key);
MortonKey* morton_key_from_anchor(KeyType (*anchor)[3]);
MortonKey* morton_key_from_morton(KeyType key);
MortonKey* morton_key_from_point(PointType (*point)[3]);
MortonKey* morton_key_parent(MortonKey *key);
KeyType morton_key_level(MortonKey *key);
MortonKey* morton_key_first_child(MortonKey *key);
void morton_key_children(MortonKey *key, uintptr_t *ptr);
void morton_key_to_coordinates(MortonKey *morton, PointType (*origin)[3], PointType (*diameter)[3], PointType (*coord)[3]);
void morton_key_box_coordinates(MortonKey *morton, PointType (*origin)[3], PointType (*diameter)[3], PointType (*coord)[24]);
MortonKey* morton_key_key_in_direction(MortonKey *morton, int64_t (*direction)[3]);
bool morton_key_is_ancestor(MortonKey *morton, MortonKey *other);
bool morton_key_is_descendent(MortonKey *morton, MortonKey *other);
"""

ffi.cdef(types)
ffi.cdef(methods)

lib = ffi.dlopen(os.path.join(LIBDIR,  lib_name))

# Deepest Level
DEEPEST_LEVEL = 16

# The level displacement in a Morton index in bits
LEVEL_DISPLACEMENT = 15
