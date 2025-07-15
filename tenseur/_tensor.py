from __future__ import absolute_import

import tenseurbackend as backend

from tenseurbackend import data_type as dtype
from tenseurbackend import vector_shape, matrix_shape, tensor3_shape, tensor4_shape, tensor5_shape

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

def _from_numpy_data_type(data_type):
  if data_type == np.float32:
    return dtype.float32
  elif data_type == np.float64:
    return dtype.float64
  else:
    raise RuntimeError("Data type not supported.")

def _to_numpy_data_type(data_type):
  if data_type == dtype.float32:
    return np.float32
  elif data_type == dtype.float64:
    return np.float64
  else:
    raise RuntimeError("Data type not supported.")

"""
Create a tensor from shape (rank) and data type
"""
class tensor(object):

  def __init__(self, dims, data_type = dtype.float32, data = "auto"):
    self.dims = _make_tuple_shape(dims)
    self.dims_rank = len(self.dims)
    assert(self.dims_rank >0 and self.dims_rank <= 5)
    self.data_type = data_type
    #assert(data == "auto" or data == None)
    self.t = _get_tensor(data_type, self.dims_rank, self.dims, data)

  def dtype(self):
    return self.data_type

  def data(self):
    return self.t

  def rank(self):
    return self.dims_rank

  def size(self):
    return self.t.size()

  def shape(self):
    return self.t.shape()

  def strides(self):
    return self.t.strides()

  def __getitem__(self, index):
    if index >= self.size():
      raise StopIteration()
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
    return tensor(self.dims, c.data_type(), c)

  def __sub__(self, other):
    assert(self.rank() == other.rank())
    assert(self.size() == other.size())
    c = (self.t - other.t).eval()
    return tensor(self.dims, c.data_type(), c)

  def __mul__(self, other):
    c = (self.t * other.t).eval()
    if self.rank() == 1 and other.rank() == 1:
      return tensor(self.dims, c.data_type(), c)
    elif self.rank() == 2 and other.rank() == 1:
      return tensor((self.dims[0]), c.data_type(), c)
    elif self.rank() == 2 and other.rank() == 2:
      return tensor((self.dims[0], other.dims[1]), c.data_type(), c)
    else:
      raise RuntimeError("Multiplication not supported.")

  def __truediv__(self, other):
    assert(self.rank() == other.rank())
    assert(self.size() == other.size())
    c = (self.t / other.t).eval()
    return tensor(self.dims, c.data_type(), c)

  def __matmul__(self, other):
    return self.__mul__(other)

  def __repr__(self):
    return repr(self.t)

  def copy(self):
    return tensor(self.dims, self.data_type, self.t.copy())

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
def vector(size, data_type = dtype.float32):
  assert(isinstance(size, int))
  return tensor(size, data_type)

"""
Create a matrix from shape and optional data type
"""
def matrix(rows, cols, data_type = dtype.float32):
  assert(isinstance(rows, int))
  assert(isinstance(cols, int))
  return tensor((rows, cols), data_type)

"""
Create a tensor from a numpy array
"""
def from_numpy(array):
  shape = array.shape
  data_type = _from_numpy_data_type(array.dtype)
  t = tensor(shape, data_type)
  size = t.size()
  rank = len(shape)
  if rank == 1:
    # Vector
    for k in range(size):
      t[k] = array[k]
  elif rank == 2:
    # Matrix
    rows = shape[0]
    cols = shape[1]
    for i in range(rows):
      for j in range(cols):
        t.set(i, j, array[i, j])
  elif rank == 3:
    # 3d tensor
    I = shape[0]
    J = shape[1]
    K = shape[2]
    for i in range(I):
      for j in range(J):
        for k in range(K):
          t.set(i, j, k, array[i, j, k])
  elif rank == 4:
    # 4d tensor
    I = shape[0]
    J = shape[1]
    K = shape[2]
    L = shape[3]
    for i in range(I):
      for j in range(J):
        for k in range(K):
          for l in range(L):
            t.set(i, j, k, l, array[i, j, k, l])
  elif rank == 5:
    I = shape[0]
    J = shape[1]
    K = shape[2]
    L = shape[3]
    M = shape[4]
    for i in range(I):
      for j in range(J):
        for k in range(K):
          for l in range(L):
            for m in range(M):
              t.set(i, j, k, l, m, array[i, j, k, l, m])
  else:
    raise RuntimeError("Array rank not supported, only up to 5d are supported.")
  return t



def zeros(dims, data_type = dtype.float32):
  shape = _make_tuple_shape(dims)
  rank = len(shape)
  if rank == 1:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.zeros_vector_float(vector_shape(dims)))
    if data_type == dtype.dfloat64:
      return tensor(shape, data_type, backend.zeros_vector_double(vector_shape(dims)))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 2:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.zeros_matrix_float(matrix_shape(*dims)))
    if data_type == dtype.dfloat64:
      return tensor(shape, data_type, backend.zeros_matrix_double(matrix_shape(*dims)))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 3:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.zeros_tensor3_float(tensor3_shape(*dims)))
    if data_type == dtype.dfloat64:
      return tensor(shape, data_type, backend.zeros_tensor3_double(tensor3_shape(*dims)))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 4:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.zeros_tensor4_float(tensor4_shape(*dims)))
    if data_type == dtype.dfloat64:
      return tensor(shape, data_type, backend.zeros_tensor4_double(tensor4_shape(*dims)))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 5:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.zeros_tensor5_float(tensor5_shape(*dims)))
    if data_type == dtype.dfloat64:
      return tensor(shape, data_type, backend.zeros_tensor5_double(tensor5_shape(*dims)))
    else:
      raise RuntimeError("Data type not yet supported.")
  else:
    raise RuntimeError(f"Tensor of rank {rank} not supported.")

def ones(dims, data_type = dtype.float32):
  shape = _make_tuple_shape(dims)
  rank = len(shape)
  if rank == 1:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.ones_vector_float(vector_shape(dims)))
    if data_type == dtype.dfloat64:
      return tensor(shape, data_type, backend.ones_vector_double(vector_shape(dims)))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 2:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.ones_matrix_float(matrix_shape(*dims)))
    if data_type == dtype.dfloat64:
      return tensor(shape, data_type, backend.ones_matrix_double(matrix_shape(*dims)))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 3:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.ones_tensor3_float(tensor3_shape(*dims)))
    if data_type == dtype.dfloat64:
      return tensor(shape, data_type, backend.ones_tensor3_double(tensor3_shape(*dims)))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 4:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.ones_tensor4_float(tensor4_shape(*dims)))
    if data_type == dtype.dfloat64:
      return tensor(shape, data_type, backend.ones_tensor4_double(tensor4_shape(*dims)))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 5:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.ones_tensor5_float(tensor5_shape(*dims)))
    if data_type == dtype.dfloat64:
      return tensor(shape, data_type, backend.ones_tensor5_double(tensor5_shape(*dims)))
    else:
      raise RuntimeError("Data type not yet supported.")
  else:
    raise RuntimeError(f"Tensor of rank {rank} not supported.")

def fill(dims, value, data_type = dtype.float32):
  shape = _make_tuple_shape(dims)
  rank = len(shape)
  if rank == 1:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.fill_vector_float(vector_shape(dims), value))
    if data_type == dtype.dfloat64:
      return tensor(shape, data_type, backend.fill_vector_double(vector_shape(dims), value))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 2:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.fill_matrix_float(matrix_shape(*dims), value))
    if data_type == dtype.dfloat64:
      return tensor(shape, data_type, backend.fill_matrix_double(matrix_shape(*dims), value))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 3:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.fill_tensor3_float(tensor3_shape(*dims), value))
    if data_type == dtype.dfloat64:
      return tensor(shape, data_type, backend.fill_tensor3_double(tensor3_shape(*dims), value))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 4:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.fill_tensor4_float(tensor4_shape(*dims), value))
    if data_type == dtype.dfloat64:
      return tensor(shape, data_type, backend.fill_tensor4_double(tensor4_shape(*dims), value))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 5:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.fill_tensor5_float(tensor5_shape(*dims), value))
    if data_type == dtype.dfloat64:
      return tensor(shape, data_type, backend.fill_tensor5_double(tensor5_shape(*dims), value))
    else:
      raise RuntimeError("Data type not yet supported.")
  else:
    raise RuntimeError(f"Tensor of rank {rank} not supported.")

def arange(dims, value = 0., data_type = dtype.float32):
  shape = _make_tuple_shape(dims)
  rank = len(shape)
  if rank == 1:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.range_vector_float(vector_shape(dims), value))
    if data_type == dtype.dfloat64:
      return tensor(shape, data_type, backend.range_vector_double(vector_shape(dims), value))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 2:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.range_matrix_float(matrix_shape(*dims), value))
    if data_type == dtype.dfloat64:
      return tensor(shape, data_type, backend.range_matrix_double(matrix_shape(*dims), value))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 3:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.range_tensor3_float(tensor3_shape(*dims), value))
    if data_type == dtype.dfloat64:
      return tensor(shape, data_type, backend.range_tensor3_double(tensor3_shape(*dims), value))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 4:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.range_tensor4_float(tensor4_shape(*dims), value))
    if data_type == dtype.dfloat64:
      return tensor(shape, data_type, backend.range_tensor4_double(tensor4_shape(*dims), value))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 5:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.range_tensor5_float(tensor5_shape(*dims), value))
    if data_type == dtype.dfloat64:
      return tensor(shape, data_type, backend.range_tensor5_double(tensor5_shape(*dims), value))
    else:
      raise RuntimeError("Data type not yet supported.")
  else:
    raise RuntimeError(f"Tensor of rank {rank} not supported.")


"""
Save to a binary file
"""
def save(ten, filename):
  rank = ten.rank()
  data_type = ten.dtype()
  if rank == 1:
    if data_type == dtype.float32:
      backend.save_vector_float(ten.data(), filename)
    elif data_type == dtype.float64:
      backend.save_vector_double(ten.data(), filename)
    else:
      raise RuntimeError("Saving data type not yet supported.")
  else:
    raise RuntimeError(f"Tensor of rank {rank} not supported.")

