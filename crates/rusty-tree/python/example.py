import numpy as np

from rusty_tree import MPI_Comm
from rusty_tree.types.domain import Domain
from rusty_tree.distributed import DistributedTree

comm = MPI_Comm() 

points = np.random.rand(10000, 3)
# domain = Domain.from_global_points(points, comm)

tree = DistributedTree.from_global_points(points, False, comm)
print(comm.size, comm.rank, len(tree.keys), tree.nkeys)