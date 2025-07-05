import unittest

from test import support

import tenseur as ten

class VectorTest(unittest.TestCase):

  def setUp(self):
    self.n = 10
    self.x = ten.vector(self.n)
    self.y = ten.vector(self.n)

  def test_add(self):
    z = self.x + self.y
    a = self.x.numpy()
    b = self.y.numpy()
    c = a + b
    for i in range(self.n):
      assert(c[i] == z[i])

  def test_sub(self):
    z = self.x - self.y
    a = self.x.numpy()
    b = self.y.numpy()
    c = a - b
    for i in range(self.n):
      assert(c[i] == z[i])

  def test_mul(self):
    z = self.x * self.y
    a = self.x.numpy()
    b = self.y.numpy()
    c = a * b
    for i in range(self.n):
      assert(c[i] == z[i])

  def test_div(self):
    z = self.x / self.y
    a = self.x.numpy()
    b = self.y.numpy()
    c = a / b
    for i in range(self.n):
      assert(c[i] == z[i])

