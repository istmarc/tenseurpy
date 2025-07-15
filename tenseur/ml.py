import tenseurbackend as backend

from tenseurbackend import data_type as dtype

from ._tensor import tensor

import matplotlib.pyplot as plt

def histogram(x: tensor, standartize = False, cumulative = False, nbins = 0):
  rank = x.rank()
  assert(rank == 1)
  size = x.size()
  data_type = x.dtype()
  if data_type == dtype.float32:
    options = backend.histogram_options(standartize, cumulative, nbins)
    h = backend.histogram_float(options)
    h.fit(x.data())
    hist, bins = h.hist()
    return tensor(hist.size(), data_type, hist), tensor(bins.size(), data_type, bins)
  elif data_type == dtype.float64:
    options = backend.histogram_options(standartize, cumulative, nbins)
    h = backend.histogram_double(options)
    h.fit(x.data())
    hist, bins = h.hist()
    return tensor(hist.size(), data_type, hist), tensor(bins.size(), data_type, bins)
  else:
    raise RuntimeError("Data type not supported for fitting histogram.")


def plot_histogram(hist, bins, **args):
  fig, ax = plt.subplots()
  ax.stairs(hist.numpy(), bins.numpy(), **args)
  plt.show()

