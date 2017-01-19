# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 12:14:35 2016

@author: Pierre H. Richemond
"""
import scipy.io
from scipy.io import loadmat
from scipy import signal
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from itertools import combinations
import os
import h5py 

from sklearn.metrics import roc_auc_score, mean_squared_error, roc_curve
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.datasets import dump_svmlight_file

DATA_FOLDER= './Data/train_1'
TRAIN_1_DATA_FOLDER_IN_SINGLE_FILE=DATA_FOLDER + "/1_101_1.mat"

#---------------------------------------------------------------#
def entropy(signal):
    '''
    function returns entropy of a signal
    signal must be a 1-D numpy array
    '''
    lensig=signal.size
    symset=list(set(signal))
    numsym=len(symset)
    propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]
    ent=np.sum([p*np.log2(1.0/p) for p in propab])
    return ent
#---------------------------------------------------------------#    

#---------------------------------------------------------------#
def ieegMatToPandasDF(path):
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    return pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0])   

def ieegMatToArray(path):
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    return ndata['data']  

#---------------------------------------------------------------#

#---------------------------------------------------------------#
def ieegSingleMetaData(path):
    mat_data = scipy.io.loadmat(path)
    data = mat_data['dataStruct']
    for i in [data, data[0], data[0][0][0], data[0][0][0][0]]:
        print((i.shape, i.size))
#---------------------------------------------------------------#        

#---------------------------------------------------------------#
def ieegGetFilePaths(directory, extension='.mat'):
    filenames = sorted(os.listdir(directory))
    files_with_extension = [directory + '/' + f for f in filenames if f.endswith(extension) and not f.startswith('.')]
    return files_with_extension
#---------------------------------------------------------------#

#---------------------------------------------------------------#
# EEG clips labeled "Preictal" (k=1) for pre-seizure data segments, 
# or "Interictal" (k-0) for non-seizure data segments.
# I_J_K.mat - the Jth training data segment corresponding to the Kth 
# class (K=0 for interictal, K=1 for preictal) for the Ith patient (there are three patients).
def ieegIsInterictal(name):  
    try:
        return float(name[-5])
    except:
        return 0.0
#---------------------------------------------------------------#
"""
ieegSingleMetaData(TRAIN_1_DATA_FOLDER_IN_SINGLE_FILE)   

x=ieegMatToPandasDF(TRAIN_1_DATA_FOLDER_IN_SINGLE_FILE)
print((x.shape, x.size))

matplotlib.rcParams['figure.figsize'] = (20.0, 20.0)
n=16
for i in range(0, n):
#     print i
    plt.subplot(n, 1, i + 1)
    plt.plot(x[i +1])
"""    
    
x=ieegMatToArray(TRAIN_1_DATA_FOLDER_IN_SINGLE_FILE)
print("Forme du tableau : ")
print(x.shape)

"""
x_ent=x.ravel()
entropy(x_ent)

freq = np.fft.fft2(x)
freq = np.abs(freq)
print (freq)

freq.shape
z=np.log(freq).ravel()
z.shape

x_std = x.std(axis=1)
# print(x_std.shape, x_std.ndim)
x_split = np.array(np.split(x_std, 100))
# print(x_split.shape)
x_mean = np.mean(x_split, axis=0)
# print(x_mean.shape)

plt.subplot(3, 1, 1)
plt.plot(x)
plt.subplot(3, 1, 2)
plt.plot(x_std)
plt.subplot(3, 1, 3)
plt.plot(x_mean)
"""

#w, h = signal.freqz(taps, [1], worN=2000)
# Resample to 200Hz ( factor 2 ; 30 taps ; axis 0 ; with group delay compensation )
# x_decimated = signal.decimate(x, 2, 30, 'fir', 0 , true )
base_numtaps = 21
fs = 400

delta_fir = signal.remez(base_numtaps, [0, 4, 4+2 , 0.5*fs], [1, 0], Hz=fs)
theta_fir = signal.remez(base_numtaps, [0, 4-2 , 4, 8, 8+3 , 0.5*fs], [0, 1, 0], Hz=fs)
alpha_fir = signal.remez(base_numtaps, [0, 8-3 ,8, 12, 12+4 ,0.5*fs], [0, 1, 0], Hz=fs)
beta_fir = signal.remez(base_numtaps, [0, 12-4 ,12, 30,30+5 ,0.5*fs], [0, 1, 0], Hz=fs)
lowgamma_fir = signal.remez(base_numtaps, [0, 30-5, 30, 70, 70+10 , 0.5*fs], [0, 1, 0], Hz=fs)
highgamma_fir = signal.remez(base_numtaps, [0, 70-10, 70, 0.5*fs], [0, 1], Hz=fs)

[w, Hdelta ] = signal.freqz(delta_fir, worN = 2048)
[w, Hdelta ] = signal.freqz(theta_fir, worN = 2048)
[w, Hdelta ] = signal.freqz(alpha_fir, worN = 2048)
[w, Hdelta ] = signal.freqz(beta_fir, worN = 2048)
[w, Hdelta ] = signal.freqz(lowgamma_fir, worN = 2048)
[w, Hdelta ] = signal.freqz(highgamma_fir, worN = 2048)

length, channels = x.shape
stride = 10 * channels
epsilon = 1E-8
strider = length // stride
correlvector = np.zeros((strider , channels))
windowedx = np.vsplit(x, strider)

cwtscales = 32
for j in range(channels):
    cwtsignal = signal.cwt(x_decimated[], signal.ricker, np.arange(1, cwtscales -1) )
    plt.imshow(cwtsignal, extent=[-1, 1, 1, cwtscales -1], cmap='PRGn', aspect='auto',  
               vmax=abs(cwtsignal).max(), vmin=-abs(cwtsignal).max())

# Here we create a small perturbation tensor to account for 0 signal intervals
#They blow up the correl calc on division by 0 in the variance bit
for i in range(strider):
    corrvect = np.linalg.eigvals(np.corrcoef(windowedx[i]+epsilon*np.random.rand(stride, channels) , rowvar=0) )
    correlvector[i] = np.transpose( corrvect )
