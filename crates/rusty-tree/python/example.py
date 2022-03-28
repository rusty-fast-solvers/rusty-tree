import numpy as np

from rusty_tree import MPI_Comm
from rusty_tree.types.domain import Domain

comm = MPI_Comm() 

points = np.random.rand(1000, 3)

domain = Domain.from_global_points(points, comm.raw)

print(comm.size, comm.rank, domain.origin)