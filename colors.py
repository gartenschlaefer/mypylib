# --
# color stuff

import numpy as np


def colors_rgb_to_xyz(rgb, norm=True):
  """
  transforms a color from rgb to xyz
  y is luminance match for humans
  """

  # transformation matrix
  A = (1 / 0.17697) * (np.array([[0.49, 0.31, 0.20], [0.17697, 0.8124, 0.01063], [0, 0.01, 0.99]]))

  # rgb
  xyz = A @ rgb

  # no norm
  if not norm: return xyz

  # norm
  xyz /= np.sum(xyz) 

  return xyz



if __name__ == '__main__':
  """
  main
  """

  # input color
  rgb = np.array([0.1, 0.2, 0.05])

  # color transform
  xyz = colors_rgb_to_xyz(rgb)

  print(rgb)
  print(xyz)