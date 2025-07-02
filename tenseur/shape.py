import tenseurbackend as backend

from typing import TypeVar, Generic

class shape():
    """
    rank: Rank of the tensor
    dims: backend.list type
    """
    def __init__(self, rank, dims):
      self.rank = rank
      if rank == 1:
        self.s = backend.vector_shape
      elif rank == 2:
        self.s = backend.matrix_shape
      elif rank == 3:
        self.s == backend.tensor3_shape
      elif rank == 4:
        self.s == backend.tensor4_shape
      elif rank == 5:
        self.s = backend.tensor5_shape
      else:
        raise RuntimeError("Tensor shape rank must be between 1 and 5.")

    def __repr__(self):
      return self.s.__repr__()

