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

# ffi.cdef("""
# typedef %(_mpi_comm_t)s MPI_Comm;
# void sayhello(MPI_Comm);
# void cleanup(MPI_Comm);
# """ % vars()
# )

lib = ffi.dlopen(os.path.join(LIBDIR,  lib_name))

sayhello = lib.sayhello
cleanup = lib.cleanup

# Deepest Level
DEEPEST_LEVEL = 16

# The level displacement in a Morton index in bits
LEVEL_DISPLACEMENT = 15
