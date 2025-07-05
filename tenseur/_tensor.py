import tenseurbackend as backend

from tenseurbackend import data_type as dtype

import numpy as np

# Reduce / fold
from functools import reduce
import operator

"""
Get the tensor type (vector, matrix, tensor3, tensor4 or tensor5)
"""
def _get_tensor(data_type, rank, shape, data = "auto"):
  if data == None:
    return None
  if data != "auto":
    assert(data_type == data.data_type())
    # TODO Check rank
    assert(data.size() == reduce(operator.mul, shape))
    return data
  if rank == 1:
    if data_type == dtype.float32:
      return backend.vector_float(shape[0])
    elif data_type == dtype.float64:
      return backend.vector_double(shape[0])
    else:
      raise RuntimeError("Data type not yet supported.")
  elif rank == 2:
    if data_type == dtype.float32:
      return backend.matrix_float(shape[0], shape[1])
    elif data_type == dtype.float64:
      return backend.matrix_double(shape[0], shape[1])
    else:
      raise RuntimeError("Data type not yet supported.")
  elif rank == 3:
    if data_type == dtype.float32:
      return backend.tensor3_float(shape[0], shape[1], shape[2])
    elif data_type == dtype.float64:
      return backend.tensor3_double(shape[0], shape[1], shape[2])
    else:
      raise RuntimeError("Data type not yet supported.")
  elif rank == 4:
    if data_type == dtype.float32:
      return backend.tensor4_float(shape[0], shape[1], shape[2], shape[3])
    elif data_type == dtype.float64:
      return backend.tensor4_double(shape[0], shape[1], shape[2], shape[3])
    else:
      raise RuntimeError("Data type not yet supported.")
  elif rank == 5:
    if data_type == dtype.float32:
      return backend.tensor5_float(shape[0], shape[1], shape[2], shape[3], shape[4])
    elif data_type == dtype.float64:
      return backend.tensor5_double(shape[0], shape[1], shape[2], shape[3], shape[4])
    else:
      raise RuntimeError("Data type not yet supported.")
  else:
    raise RuntimeError("tensor support only up to 5 dimensions.")

def _make_tuple_shape(dims):
  if isinstance(dims, int):
    return tuple([dims])
  else:
    return tuple(dims)


def _to_numpy_data_type(data_type):
  if data_type == dtype.float32:
    return np.float32
  elif data_type == dtype.float64:
    return np.float64
  else:
    raise RuntimeError("Data type not supported.")

"""
Create a tensor from rank, shape and data type
"""
class tensor(object):

  def __init__(self, dims_rank, dims, data_type = dtype.float32, data = "auto"):
    assert(dims_rank >0 and dims_rank <= 5)
    self.dims_rank = dims_rank
    self.dims = _make_tuple_shape(dims)
    self.data_type = data_type
    #assert(data == "auto" or data == None)
    self.t = _get_tensor(data_type, dims_rank, self.dims, data)

  def rank(self):
    return self.dims_rank

  def size(self):
    return self.t.size()

  def shape(self):
    return self.t.shape()

  def strides(self):
    return self.t.strides()

  def __getitem__(self, index):
    return self.t[index]

  def __setitem__(self, index, value):
    self.t.__setitem__(index, value)

  def __call__(self, *index):
    return self.t.__call__(*index)

  def set(self, *index_and_value):
    self.t.set(*index_and_value)

  def __add__(self, other):
    assert(self.rank() == other.rank())
    assert(self.size() == other.size())
    c = (self.t + other.t).eval()
    return tensor(self.rank(), self.dims, c.data_type(), c)

  def __sub__(self, other):
    assert(self.rank() == other.rank())
    assert(self.size() == other.size())
    c = (self.t - other.t).eval()
    return tensor(self.rank(), self.dims, c.data_type(), c)

  def __mul__(self, other):
    c = (self.t * other.t).eval()
    if self.rank() == 1 and other.rank() == 1:
      return tensor(self.rank(), self.dims, c.data_type(), c)
    elif self.rank() == 2 and other.rank() == 1:
      return tensor(other.rank(), (self.dims[0]), c.data_type(), c)
    elif self.rank() == 2 and other.rank() == 2:
      return tensor(self.rank(), (self.dims[0], other.dims[1]), c.data_type(), c)
    else:
      raise RuntimeError("Multiplication not supported.")

  def __truediv__(self, other):
    assert(self.rank() == other.rank())
    assert(self.size() == other.size())
    c = (self.t / other.t).eval()
    return tensor(self.rank(), self.dims, c.data_type(), c)

  def __matmul__(self, other):
    return self.__mul__(other)

  def __repr__(self):
    return repr(self.t)

  def copy(self):
    return tensor(self.dims_rank, self.dims, self.data_type,
      self.t.copy())

  """
  Convert to numpy ndarray
  """
  def numpy(self):
    np_data_type = _to_numpy_data_type(self.data_type)
    array = np.zeros(self.dims, dtype = np_data_type)
    size = self.shape().size()
    if self.dims_rank == 1:
      # Vector
      for k in range(size):
        array[k] = self.t[k]
    elif self.dims_rank == 2:
      # Matrix
      rows = self.shape().dim(0)
      cols = self.shape().dim(1)
      for i in range(rows):
        for j in range(cols):
          array[i, j] = self.t(i, j)
    elif self.dims_rank == 3:
      # 3d tensor
      I = self.shape().dim(0)
      J = self.shape().dim(1)
      K = self.shape().dim(2)
      for i in range(I):
        for j in range(J):
          for k in range(K):
            array[i, j, k] = self.t(i, j, k)
    elif self.dims_rank == 4:
      # 4d tensor
      I = self.shape().dim(0)
      J = self.shape().dim(1)
      K = self.shape().dim(2)
      L = self.shape().dim(3)
      for i in range(I):
        for j in range(J):
          for k in range(K):
            for l in range(L):
              array[i, j, k, l] = self.t(i, j, k, l)
    else:
      I = self.shape().dim(0)
      J = self.shape().dim(1)
      K = self.shape().dim(2)
      L = self.shape().dim(3)
      M = self.shape().dim(4)
      for i in range(I):
        for j in range(J):
          for k in range(K):
            for l in range(L):
              for m in range(M):
                array[i, j, k, l, m] = self.t(i, j, k, l, m)
    return array

"""
Create a vector from shape and optional data type
"""
def vector(dims, data_type = dtype.float32):
  return tensor(1, dims, data_type)

"""
Create a matrix from shape and optional data type
"""
def matrix(dims, data_type = dtype.float32):
  return tensor(2, dims, data_type)

