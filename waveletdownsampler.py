# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 12:20:01 2016

Parallel downsampling with wavelets

@author: Pierre H. Richemond
"""

import pandas as pd
import numpy as np
from scipy.io import loadmat
import argparse
import glob
import os
import pywt
import sys
import multiprocessing as mp
from collections import Counter


def write_classes(files, output_dir):
  '''Write classes to csv file'''
  classes = [int(os.path.basename(f).split('.')[0].split('_')[-1]) for f in files]
  print('Class summary: {}'.format(Counter(classes)))
  df = pd.DataFrame(data=classes)
  df.to_csv(os.path.join(output_dir, 'classes.csv'))


def wavelet_transform(df, wavelet_level, wavelet_type):
  '''Perform wavelet transform on dataframe columns'''
  arr = []
  for col in df.columns:
    tmp = pywt.wavedec(df.ix[:, col], wavelet_type, level=wavelet_level)[0]
    arr.append(np.reshape(tmp, [-1, 1]))
  return np.concatenate(arr, axis=1)


def read_mat(path):
  '''Read data from mat file into pandas dataframe'''
  mat = loadmat(path)
  names = mat['dataStruct'].dtype.names
  ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
  return pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0])


def convert_dataframe(data, wavelet_level=None, wavelet_type='db12'):
  '''Convert dataframe to numpy array, optionally performing wavelet transform'''
  if wavelet_level:
    data = wavelet_transform(data, wavelet_level, wavelet_type)
  return np.array(data, dtype=np.float32)


def write_matrix(filepath, output_dir, wavelet_level=None, wavelet_type='db12'):
  '''Write the numpy array to binary format'''
  newfile = os.path.basename(filepath).split('.')[0] + '.npy'
  newpath = os.path.join(output_dir, newfile)
  if os.path.exists(newpath):
    print('Skipping: {}'.format(newpath))
    return None
  else:
    data = read_mat(filepath)
    data = convert_dataframe(data, wavelet_level, wavelet_type) 
    with open(newpath, 'w') as f:
      np.save(f, data)
    print('Wrote: {}'.format(newpath))


def write_matrices(args):
  '''Wrapper function for multiprocessing call'''
  try:
    result = write_matrix(args[0], args[1], args[2], args[3])
  except Exception as e:
    print('FAILED: {}'.format(args[0]))
    print('  with error: {}'.format(e))


def main(input_dir, output_dir, wavelet_level=None, wavelet_type='db12', ncores=None):
  
  # Try to create the output directory if it does not exist.
  try:
    os.makedirs(output_dir)
  except:
    print('Directory "{}" already exists, skipping'.format(output_dir))

  # Glob the mat files in the target directory.
  files = glob.glob(os.path.join(input_dir, '*.mat'))
  print('Processing {} files'.format(len(files)))

  # Write a csv of classes.
  write_classes(files, output_dir)

  # Create a work item list for multiprocessing pool to munch on.
  args = [[f, output_dir, wavelet_level, wavelet_type] for f in files]
  
  # If more than 1 core is requested, use multiprocessing.
  if ncores > 1:
    pool = mp.Pool()
    try:
      _ = pool.map(write_matrices, args)
    except:
      print('Multiprocessing ERROR')
    pool.close()
    pool.join()
  else:
    _ = [write_matrices(x) for x in args]


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('input_dir', type=str, default=None)
  parser.add_argument('output_dir', type=str, default=None)
  parser.add_argument('--wavelet_level', type=int, default=None)
  parser.add_argument('--wavelet_type', type=str, default='db12')
  parser.add_argument('--ncores', type=int, default=1)

  args = parser.parse_args()

  main(args.input_dir, args.output_dir, args.wavelet_level, args.wavelet_type, args.ncores)