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

if MPI._sizeof(MPI.Comm) == ffi.sizeof('int'):
    _mpi_comm_t = 'int'
else:
    _mpi_comm_t = 'void*'

types =f"""
typedef uint64_t KeyType;
typedef double PointType;
typedef {_mpi_comm_t} MPI_Comm;

typedef struct {{
    KeyType anchor[3];
    KeyType morton;
}} MortonKey;

typedef struct {{
    PointType coordinate[3];
    size_t global_idx;
    MortonKey key;
}} Point;

typedef struct {{
    PointType origin[3];
    PointType diameter[3];
}} Domain;

typedef struct {{
    bool balanced;
    Point (*points)[];
    MortonKey (*keys)[];
}} DistributedTree;
"""

morton = """
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

# distributed = """
# Tree* tree_from_morton_keys(KeyType *data, size_t len);
# """

domain = """
Domain* domain_from_local_points(PointType (*point)[][3], size_t len);
Domain* domain_from_global_points(PointType (*point)[][3], size_t len, MPI_Comm comm);
"""

constants = """
extern size_t LEVEL_DISPLACEMENT;
extern KeyType LEVEL_MASK;
extern KeyType BYTE_MASK;
extern KeyType BYTE_DISPLACEMENT;
extern KeyType NINE_BIT_MASK;
extern int64_t DIRECTIONS[26][3];
extern KeyType Z_LOOKUP_ENCODE[256];
extern KeyType Z_LOOKUP_DECODE[512];
extern KeyType Y_LOOKUP_ENCODE[256];
extern KeyType Y_LOOKUP_DECODE[512];
extern KeyType X_LOOKUP_ENCODE[256];
extern KeyType X_LOOKUP_DECODE[512];
"""

mpi = """
void cleanup(MPI_Comm);
"""

ffi.cdef(types)
ffi.cdef(morton)
ffi.cdef(domain)
ffi.cdef(constants)

lib = ffi.dlopen(os.path.join(LIBDIR,  lib_name))


# The level displacement in a Morton index in bits
LEVEL_DISPLACEMENT = 15


class MPI_Comm:
    """Interface for raw/wrapped communicator"""
    def __init__(self, comm):
        self.comm = comm
        self.ptr = MPI._addressof(self.comm)
        self.val = ffi.cast('MPI_Comm*', self.ptr)[0]