import tenseurbackend as backend

from tenseurbackend import data_type as dtype

from typing import TypeVar, Generic, Type

import numpy as np

"""
Get the tensor type (vector, matrix, tensor3, tensor4 or tensor5)
"""
def _get_tensor(data_type, rank, shape):
  if rank == 1:
    if data_type == dtype.dfloat:
      return backend.vector_float(shape)
    elif data_type == dtype.ddouble:
      return backend.vector_double(shape)
    else:
      raise RuntimeError("Data type not yet supported.")
  elif rank == 2:
    if data_type == dtype.dfloat:
      return backend.matrix_float(shape[0], shape[1])
    elif data_type == dtype.ddouble:
      return backend.matrix_double(shape[0], shape[1])
    else:
      raise RuntimeError("Data type not yet supported.")
  elif rank == 3:
    if data_type == dtype.dfloat:
      return backend.tensor3_float(shape[0], shape[1], shape[2])
    elif data_type == dtype.ddouble:
      return backend.tensor3_double(shape[0], shape[1], shape[2])
    else:
      raise RuntimeError("Data type not yet supported.")
  elif rank == 4:
    if data_type == dtype.dfloat:
      return backend.tensor4_float(shape[0], shape[1], shape[2], shape[3])
    elif data_type == dtype.ddouble:
      return backend.tensor4_double(shape[0], shape[1], shape[2], shape[3])
    else:
      raise RuntimeError("Data type not yet supported.")
  elif rank == 5:
    if data_type == dtype.dfloat:
      return backend.tensor5_float(shape[0], shape[1], shape[2], shape[3], shape[4])
    elif data_type == dtype.ddouble:
      return backend.tensor5_double(shape[0], shape[1], shape[2], shape[3], shape[4])
    else:
      raise RuntimeError("Data type not yet supported.")
  else:
    raise RuntimeError("tensor support only up to 5 dimensions.")

"""
Create a tensor from rank, shape and data type
"""
class tensor(object):

  def __init__(self, rank, dims, data_type = dtype.dfloat):
    assert(rank >0 and rank <= 5)
    self.rank = rank
    self.dims = dims
    self.data_type = data_type
    self.t = _get_tensor(data_type, rank, dims)

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

  def __repr__(self):
    return repr(self.t)

  def copy(self):
    s = tensor(self.rank, self.dims, self.data_type)
    for i in range(self.shape().size()):
      s[i] = self.t[i]
    return s

  """
  Convert to numpy ndarray
  """
  def numpy(self):
    if self.dims is int:
      dims = [self.dims]
    else:
      dims = self.dims
    array = np.zeros(dims)
    size = self.shape().size()
    if rank == 1:
      # Vector
      for k in range(size):
        array[k] = self.t[k]
    elif rank == 2:
      # Matrix
      rows = self.shape().dim(0)
      cols = self.shape().dim(1)
      for i in range(rows):
        for j in range(cols):
          array[rows, cols] = self.t(rows, cols)
    elif rank == 3:
      # TODO 3d tensor
      pass
    elif rank == 4
      # TODO 4d tensor
      pass
    else:
      pass
      # TODO 5d tensor
    return array

"""
Create a vector from shape and optional data type
"""
def vector(dims, data_type = dtype.dfloat):
  return tensor(1, dims, data_type)

"""
Create a matrix from shape and optional data type
"""
def matrix(dims, data_type = dtype.dfloat):
  return tensor(2, dims, data_type)

