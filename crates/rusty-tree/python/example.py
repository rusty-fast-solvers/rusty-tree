import numpy as np

from rusty_tree import MPI_Comm
from rusty_tree.distributed import DistributedTree

comm = MPI_Comm()

points = np.random.rand(1000, 3)

tree = DistributedTree.from_global_points(points, True, comm)
print(comm.size, comm.rank, tree.nkeys)