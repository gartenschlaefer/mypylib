# --
# cv filters

import numpy as np
from skimage.util.shape import view_as_windows
from scipy.signal import convolve2d


def zero_crossing(img):
  """
  todo
  """
  pass
  

def my_gaussian_filter(img, r=5, sigma=1):
  """
  gaussian filtering
  """

  # var
  x = np.arange(-r//2+1, r//2+1)
  mu = 0

  print(x)

  # gaussian function
  y = np.array([(1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(1 / 2) * ((x - mu) / sigma)**2)])

  # kernel
  k = np.outer(y, y)
  #k = np.einsum('ij,ji->ij', y, y)
  #print(k)
  #stop

  # convolve
  img_filt = convolve2d(img, k, mode='same', boundary='symm', fillvalue=0)

  #img_filt2 = convolve2d(img, y, mode='same', boundary='symm', fillvalue=0)
  #img_filt2 = convolve2d(img_filt2, y, mode='same', boundary='symm', fillvalue=0)

  #return img_filt, img_filt2
  return img_filt



def my_convolve2d(img, k, pad_mode='symmetric'):
  """
  my convolution in 2d
  """

  # shape
  m, n = k.shape

  # image padding
  img_pad = np.pad(img, (m//2, n//2), mode=pad_mode)

  # parts
  img_win = view_as_windows(img_pad, window_shape=k.shape, step=(1, 1))

  # einsum
  img_conv = np.einsum('mnjk,jk->mn', img_win, k)

  # reshape
  img_conv = img_conv.reshape(img.shape)

  return img_conv


def my_sobel_filter(img):
  """
  simple sobel filter
  """

  # kernel
  k_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  k_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
  img_x = convolve2d(img, k_x, mode='same', boundary='symm', fillvalue=0)
  img_y = convolve2d(img, k_y, mode='same', boundary='symm', fillvalue=0)
  return (img_x, img_y)


def my_roberts_filter(img):
  """
  simple roberts filter
  """

  # kernel
  k_x = np.array([[0, 1], [-1, 0]])
  k_y = np.array([[1, 0], [0, -1]])
  img_x = convolve2d(img, k_x, mode='same', boundary='symm', fillvalue=0)
  img_y = convolve2d(img, k_y, mode='same', boundary='symm', fillvalue=0)
  return (img_x, img_y)


def my_laplace_filter(img):
  """
  simple laplace filter
  """

  # kernel
  k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

  # convolve
  #img_l = convolve2d(img, k, mode='full', boundary='symm', fillvalue=0)
  #img_l = convolve2d(img, k, mode='same', boundary='fill', fillvalue=0)
  img_l = convolve2d(img, k, mode='same', boundary='symm', fillvalue=0)

  # my conv
  #img_l = my_convolve2d(img, k, pad_mode='symmetric')

  return img_l


if __name__ == '__main__':
  """
  main
  """

  # get image
  img = np.pad(np.arange(3*3).reshape(3, 3), (3, 3))
  print(img), print(img.shape)

  # # filter image
  # img_l = my_laplace_filter(img)
  # print("laplace: "), print(img_l), print(img_l.shape)
  # img_x, img_y = my_sobel_filter(img)
  # print("sobel x: "), print(img_x), print(img_x.shape)
  # print("sobel y: "), print(img_y), print(img_y.shape)
  # img_x, img_y = my_roberts_filter(img)
  # print("sobel x: "), print(img_x), print(img_x.shape)
  # print("sobel y: "), print(img_y), print(img_y.shape)

  # gaussian filter
  img_filt = my_gaussian_filter(img, r=5, sigma=1)
  print("filt: "), print(img_filt), print(img_filt.shape)