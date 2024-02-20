# --
# math functions like gaussians etc.

import numpy as np
from plot import plot1d


def gaussian(mu=0.0, sigma=1.0, samples=100):
  """
  gaussian function
  """
  x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, samples)
  y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2) 
  return x, y


if __name__ == '__main__':
  """
  main
  """

  # gaussian
  x, y = gaussian()
  print(y)
  print(np.sum(y) * (x[1] - x[0]))
  plot1d(x, y)