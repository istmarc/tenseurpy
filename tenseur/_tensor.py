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
def _get_tensor(data_type, rank, shape, requires_grad, data = "auto"):
  if data == None:
    return None
  if data != "auto":
    assert(data_type == data.data_type())
    # TODO Check rank
    assert(data.size() == reduce(operator.mul, shape))
    return data
  if rank == 1:
    if data_type == dtype.float32:
      return backend.vector_float(vector_shape(shape[0]), requires_grad)
    elif data_type == dtype.float64:
      return backend.vector_double(vector_shape(shape[0]), requires_grad)
    else:
      raise RuntimeError("Data type not yet supported.")
  elif rank == 2:
    if data_type == dtype.float32:
      return backend.matrix_float(matrix_shape(shape[0], shape[1]), requires_grad)
    elif data_type == dtype.float64:
      return backend.matrix_double(matrix_shape(shape[0], shape[1]), requires_grad)
    else:
      raise RuntimeError("Data type not yet supported.")
  elif rank == 3:
    if data_type == dtype.float32:
      return backend.tensor3_float(tensor3_shape(shape[0], shape[1], shape[2]), requires_grad)
    elif data_type == dtype.float64:
      return backend.tensor3_double(tensor3_shape(shape[0], shape[1], shape[2]), requires_grad)
    else:
      raise RuntimeError("Data type not yet supported.")
  elif rank == 4:
    if data_type == dtype.float32:
      return backend.tensor4_float(tensor4_shape(shape[0], shape[1], shape[2], shape[3]), requires_grad)
    elif data_type == dtype.float64:
      return backend.tensor4_double(tensor4_shape(shape[0], shape[1], shape[2], shape[3]), requires_grad)
    else:
      raise RuntimeError("Data type not yet supported.")
  elif rank == 5:
    if data_type == dtype.float32:
      return backend.tensor5_float(tensor5_shape(shape[0], shape[1], shape[2], shape[3], shape[4]), requires_grad)
    elif data_type == dtype.float64:
      return backend.tensor5_double(tensor5_shape(shape[0], shape[1], shape[2], shape[3], shape[4]), requires_grad)
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
Create a tensor with optional gradient information from shape (rank) and data type
"""
class tensor(object):

  """
    tensor(dims, requires_grad, data_type, data = "auto")
    Create a tensor with optional gradient information
  """
  def __init__(self, dims, requires_grad = False, data_type = dtype.float32, data = "auto"):
    self.dims = _make_tuple_shape(dims)
    self.requires_grad = requires_grad
    self.dims_rank = len(self.dims)
    assert(self.dims_rank >0 and self.dims_rank <= 5)
    self.data_type = data_type
    #assert(data == "auto" or data == None)
    self.t = _get_tensor(data_type, self.dims_rank, self.dims, requires_grad, data)

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

  def requires_grad(self):
    return self.t.requires_grad()

  def grad(self):
    return self.t.grad();

  def __getitem__(self, index):
    if (isinstance(index, int)):
      if index >= self.size():
        raise StopIteration()
      return self.t[index]
    else:
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
    return self.t + other.t

  def __sub__(self, other):
    assert(self.rank() == other.rank())
    assert(self.size() == other.size())
    return self.t - other.t

  def __mul__(self, other):
    return self.t * other.t

  def __truediv__(self, other):
    assert(self.rank() == other.rank())
    assert(self.size() == other.size())
    return self.t / other.t

  def __matmul__(self, other):
    return self.__mul__(other)

  def __repr__(self):
    return repr(self.t)

  # FIXME Copy gradient info, to see on the backend
  def copy(self):
    return tensor(self.dims, self.requires_grad, self.data_type, self.t.copy())

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
def vector(size, requires_grad = False, data_type = dtype.float32):
  assert(isinstance(size, int))
  return tensor((size), requires_grad, data_type)

"""
Create a matrix from shape and optional data type
"""
def matrix(rows, cols, requires_grad = False, data_type = dtype.float32):
  assert(isinstance(rows, int))
  assert(isinstance(cols, int))
  return tensor((rows, cols), requires_grad, data_type)

"""
Create a tensor from a numpy array
"""
def from_numpy(array, requires_grad = False):
  shape = array.shape
  data_type = _from_numpy_data_type(array.dtype)
  t = tensor(shape, requires_grad, data_type)
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


"""
Returns a tensor of zeros of data_type
"""
def zeros(dims, data_type = dtype.float32):
  shape = _make_tuple_shape(dims)
  rank = len(shape)
  if rank == 1:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.zeros_vector_float(vector_shape(dims)))
    if data_type == dtype.float64:
      return tensor(shape, data_type, backend.zeros_vector_double(vector_shape(dims)))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 2:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.zeros_matrix_float(matrix_shape(*dims)))
    if data_type == dtype.float64:
      return tensor(shape, data_type, backend.zeros_matrix_double(matrix_shape(*dims)))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 3:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.zeros_tensor3_float(tensor3_shape(*dims)))
    if data_type == dtype.float64:
      return tensor(shape, data_type, backend.zeros_tensor3_double(tensor3_shape(*dims)))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 4:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.zeros_tensor4_float(tensor4_shape(*dims)))
    if data_type == dtype.float64:
      return tensor(shape, data_type, backend.zeros_tensor4_double(tensor4_shape(*dims)))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 5:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.zeros_tensor5_float(tensor5_shape(*dims)))
    if data_type == dtype.float64:
      return tensor(shape, data_type, backend.zeros_tensor5_double(tensor5_shape(*dims)))
    else:
      raise RuntimeError("Data type not yet supported.")
  else:
    raise RuntimeError(f"Tensor of rank {rank} not supported.")

"""
Returns a tensor of ones of data type
"""
def ones(dims, data_type = dtype.float32):
  shape = _make_tuple_shape(dims)
  rank = len(shape)
  if rank == 1:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.ones_vector_float(vector_shape(dims)))
    if data_type == dtype.float64:
      return tensor(shape, data_type, backend.ones_vector_double(vector_shape(dims)))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 2:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.ones_matrix_float(matrix_shape(*dims)))
    if data_type == dtype.float64:
      return tensor(shape, data_type, backend.ones_matrix_double(matrix_shape(*dims)))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 3:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.ones_tensor3_float(tensor3_shape(*dims)))
    if data_type == dtype.float64:
      return tensor(shape, data_type, backend.ones_tensor3_double(tensor3_shape(*dims)))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 4:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.ones_tensor4_float(tensor4_shape(*dims)))
    if data_type == dtype.float64:
      return tensor(shape, data_type, backend.ones_tensor4_double(tensor4_shape(*dims)))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 5:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.ones_tensor5_float(tensor5_shape(*dims)))
    if data_type == dtype.float64:
      return tensor(shape, data_type, backend.ones_tensor5_double(tensor5_shape(*dims)))
    else:
      raise RuntimeError("Data type not yet supported.")
  else:
    raise RuntimeError(f"Tensor of rank {rank} not supported.")

"""
Returns a tensor filled with value of data_type
"""
def fill(dims, value, data_type = dtype.float32):
  shape = _make_tuple_shape(dims)
  rank = len(shape)
  if rank == 1:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.fill_vector_float(vector_shape(dims), value))
    if data_type == dtype.float64:
      return tensor(shape, data_type, backend.fill_vector_double(vector_shape(dims), value))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 2:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.fill_matrix_float(matrix_shape(*dims), value))
    if data_type == dtype.float64:
      return tensor(shape, data_type, backend.fill_matrix_double(matrix_shape(*dims), value))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 3:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.fill_tensor3_float(tensor3_shape(*dims), value))
    if data_type == dtype.float64:
      return tensor(shape, data_type, backend.fill_tensor3_double(tensor3_shape(*dims), value))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 4:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.fill_tensor4_float(tensor4_shape(*dims), value))
    if data_type == dtype.float64:
      return tensor(shape, data_type, backend.fill_tensor4_double(tensor4_shape(*dims), value))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 5:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.fill_tensor5_float(tensor5_shape(*dims), value))
    if data_type == dtype.float64:
      return tensor(shape, data_type, backend.fill_tensor5_double(tensor5_shape(*dims), value))
    else:
      raise RuntimeError("Data type not yet supported.")
  else:
    raise RuntimeError(f"Tensor of rank {rank} not supported.")

"""
Returns a range starting from value of data_type
"""
def arange(dims, value = 0., data_type = dtype.float32):
  shape = _make_tuple_shape(dims)
  rank = len(shape)
  if rank == 1:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.range_vector_float(vector_shape(dims), value))
    if data_type == dtype.float64:
      return tensor(shape, data_type, backend.range_vector_double(vector_shape(dims), value))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 2:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.range_matrix_float(matrix_shape(*dims), value))
    if data_type == dtype.float64:
      return tensor(shape, data_type, backend.range_matrix_double(matrix_shape(*dims), value))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 3:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.range_tensor3_float(tensor3_shape(*dims), value))
    if data_type == dtype.float64:
      return tensor(shape, data_type, backend.range_tensor3_double(tensor3_shape(*dims), value))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 4:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.range_tensor4_float(tensor4_shape(*dims), value))
    if data_type == dtype.float64:
      return tensor(shape, data_type, backend.range_tensor4_double(tensor4_shape(*dims), value))
    else:
      raise RuntimeError("Data type not yet supported.")
  if rank == 5:
    if data_type == dtype.float32:
      return tensor(shape, data_type, backend.range_tensor5_float(tensor5_shape(*dims), value))
    if data_type == dtype.float64:
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
  elif rank == 2:
    if data_type == dtype.float32:
      backend.save_matrix_float(ten.data(), filename)
    elif data_type == dtype.float64:
      backend.save_matrix_double(ten.data(), filename)
    else:
      raise RuntimeError("Saving data type not yet supported.")
  elif rank == 3:
    if data_type == dtype.float32:
      backend.save_tensor3_float(ten.data(), filename)
    elif data_type == dtype.float64:
      backend.save_tensor3_double(ten.data(), filename)
    else:
      raise RuntimeError("Saving data type not yet supported.")
  elif rank == 4:
    if data_type == dtype.float32:
      backend.save_tensor4_float(ten.data(), filename)
    elif data_type == dtype.float64:
      backend.save_tensor4_double(ten.data(), filename)
    else:
      raise RuntimeError("Saving data type not yet supported.")
  elif rank == 5:
    if data_type == dtype.float32:
      backend.save_tensor5_float(ten.data(), filename)
    elif data_type == dtype.float64:
      backend.save_tensor5_double(ten.data(), filename)
    else:
      raise RuntimeError("Saving data type not yet supported.")
  else:
    raise RuntimeError(f"Tensor of rank {rank} not supported.")

"""
Load from binary file
"""
def load(filename, rank, data_type = dtype.float32):
  if rank == 1:
    if data_type == dtype.float32:
      data = backend.load_vector_float(filename)
      size = data.size()
      return tensor(size, data_type, data)
    elif data_type == dtype.float64:
      data = backend.load_vector_double(filename)
      size = data.size()
      return tensor(size, data_type, data)
  else:
    raise RuntimeError(f"Tensor of rank {rank} not supported.")

"""
Returns the minimum of a tensor
"""
def min(x:tensor):
  rank = x.rank()
  data_type = x.dtype()
  if rank == 1:
    if data_type == dtype.float32:
      return backend.min_vector_float(x.data()).eval().value()
    elif data_type == dtype.float64:
      return backend.min_vector_double(x.data()).eval().value()
    else:
      raise RuntimeError("Saving data type not yet supported.")
  elif rank == 2:
    if data_type == dtype.float32:
      return backend.min_matrix_float(x.data()).eval().value()
    elif data_type == dtype.float64:
      return backend.min_matrix_double(x.data()).eval().value()
    else:
      raise RuntimeError("Saving data type not yet supported.")
  elif rank == 3:
    if data_type == dtype.float32:
      return backend.min_tensor3_float(x.data()).eval().value()
    elif data_type == dtype.float64:
      return backend.min_tensor3_double(x.data()).eval().value()
    else:
      raise RuntimeError("Saving data type not yet supported.")
  elif rank == 4:
    if data_type == dtype.float32:
      return backend.min_tensor4_float(x.data()).eval().value()
    elif data_type == dtype.float64:
      return backend.min_tensor4_double(x.data()).eval().value()
    else:
      raise RuntimeError("Saving data type not yet supported.")
  elif rank == 5:
    if data_type == dtype.float32:
      return backend.min_tensor5_float(x.data()).eval().value()
    elif data_type == dtype.float64:
      return backend.min_tensor5_double(x.data()).eval().value()
    else:
      raise RuntimeError("Saving data type not yet supported.")
  else:
    raise RuntimeError(f"Tensor of rank {rank} not supported.")

"""
Returns the maximum of a tensor
"""
def max(x:tensor):
  rank = x.rank()
  data_type = x.dtype()
  if rank == 1:
    if data_type == dtype.float32:
      return backend.max_vector_float(x.data()).eval().value()
    elif data_type == dtype.float64:
      return backend.max_vector_double(x.data()).eval().value()
    else:
      raise RuntimeError("Saving data type not yet supported.")
  elif rank == 2:
    if data_type == dtype.float32:
      return backend.max_matrix_float(x.data()).eval().value()
    elif data_type == dtype.float64:
      return backend.max_matrix_double(x.data()).eval().value()
    else:
      raise RuntimeError("Saving data type not yet supported.")
  elif rank == 3:
    if data_type == dtype.float32:
      return backend.max_tensor3_float(x.data()).eval().value()
    elif data_type == dtype.float64:
      return backend.max_tensor3_double(x.data()).eval().value()
    else:
      raise RuntimeError("Saving data type not yet supported.")
  elif rank == 4:
    if data_type == dtype.float32:
      return backend.max_tensor4_float(x.data()).eval().value()
    elif data_type == dtype.float64:
      return backend.max_tensor4_double(x.data()).eval().value()
    else:
      raise RuntimeError("Saving data type not yet supported.")
  elif rank == 5:
    if data_type == dtype.float32:
      return backend.max_tensor5_float(x.data()).eval().value()
    elif data_type == dtype.float64:
      return backend.max_tensor5_double(x.data()).eval().value()
    else:
      raise RuntimeError("Saving data type not yet supported.")
  else:
    raise RuntimeError(f"Tensor of rank {rank} not supported.")


