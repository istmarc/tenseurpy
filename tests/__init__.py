import unittest
import os
import sys
from pathlib import Path

path = Path(__file__).parent.parent

sys.path.insert(0, path)

import tenseur as ten

from tests.test_tensor import *

if __name__ == "__main__":
  unittest.main()
