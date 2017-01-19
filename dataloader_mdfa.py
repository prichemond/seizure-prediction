# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 17:51:29 2016

@author: Pierre H. Richemond
"""
import numpy as np
import scipy as sci
import scipy.signal as signal
from scipy.io import loadmat, savemat
import os
import time
import pickle
from pymdfa import quickMDFA, quickMFDXA, quickMFDCA
import matplotlib.pyplot as plt
import multiprocessing

PATIENT_NUMBER = 1

FEATURES_NAME = 'MDFA'
QSTEP = 2
QS = np.arange(-5,5.01,1.0/QSTEP)
SCSTEP = 8
SCALES = np.floor(2.0**np.arange(4,10.1,1.0/SCSTEP)).astype('i4')
WINDOW_SIZE = 60
GLOBAL_STRIDE = 36
PLOT_SPECTRA = True

RND_SEED = 1337
np.random.seed(RND_SEED)
INTERICTAL = 0
PREICTAL = 1

LOW_FREQ = 0.1
HI_FREQ = 180
SAMPLING_FREQ = 400
FILTER_B, FILTER_A = signal.butter(5, np.array([LOW_FREQ, HI_FREQ]) / (SAMPLING_FREQ / 2), btype='band')

base_dir_train = u'train_'+ str(PATIENT_NUMBER) + '/'
base_dir_tests = u'test_' + str(PATIENT_NUMBER) + '/'

def reduce_spectrum(hqc, dqc):
    # need to implement the real one...
    return np.mean(hqc), np.mean(dqc)

def compute_mdfa(x, data_length_sec, sampling_frequency, win_length_sec, stride_sec):
    n_channels = x.shape[0]
    n_timesteps = (data_length_sec - win_length_sec) / stride_sec + 1
    print(n_timesteps)
    xstacked = []

    for i in range(n_channels):
        for frame_num, w in enumerate(range(0, data_length_sec - win_length_sec + 1, stride_sec)):
                xw = x[i, w * sampling_frequency: (w + win_length_sec) * sampling_frequency]
                hqc, dqc = quickMDFA(xw, SCALES, QS)
                hmean, hstd = reduce_spectrum(hqc, dqc)
                if PLOT_SPECTRA == True:
                    plt.figure(figsize=(5,5)) 
                    plt.plot(hqc,dqc,'.-')
                    plt.title('Channel '+str(i)+' and window '+str(frame_num))
                    plt.show()
        xstacked.append(hqc)
        print(i)
            
    return np.float32(np.vstack(xstacked))

def data_preprocessor(channels_data, win_length = WINDOW_SIZE, stride = GLOBAL_STRIDE):
    filtered_data = np.float32( signal.lfilter(FILTER_B, FILTER_A, channels_data, axis=1))
    data = compute_mdfa(filtered_data, 600, SAMPLING_FREQ, win_length, stride )
    return data
    
def get_class_from_name(name):
    try:
        return float(name[-5])
    except:
        return 0.0

def get_file_names_and_classes(base_dir):
    ignored_files = ['.DS_Store', '1_45_1.mat'] 
    return np.array(
        [
            (file, get_class_from_name(file)) 
            for file in os.listdir(base_dir) if file not in ignored_files
        ],
        dtype=[('file', '|S16'), ('class', 'float32')]
    )

def get_X_from_files(base_dir, files, show_progress=True):
    X = []
    total_files = len(files)

    for i, filename in enumerate(files):
        if show_progress and i % int(total_files / 20) == 0:
            print(u'{}%: Loading file {}'.format(1+ int(i * 100 / total_files), filename))

        try:
            mat_data = loadmat(''.join([base_dir, filename.decode('UTF-8')]))
        except ValueError as ex:
            print(u'Error loading MAT file {}: {}'.format(filename, str(ex)))
            continue
        # Gets a 16x240,000 matrix => 16 channels reading data for 10 minutes at 400Hz
        channels_data = mat_data['dataStruct'][0][0][0].transpose()
        #processed_data = data_preprocessor(channels_data)
        #X = np.dstack([X, processed_data]) if X is not None else processed_data
        print(filename)        
        X.append( data_preprocessor(channels_data))
        
        
    return np.dstack(X)
    
def data_pickle(directory=base_dir_train):
        data_files_all = get_file_names_and_classes(directory)
        unique, counts = np.unique(data_files_all['class'], return_counts=True)
        occurrences = dict(zip(unique, counts))
        number_interictal = occurrences.get(INTERICTAL)
        number_preictal = occurrences.get(PREICTAL)
        print('Interictal samples:', number_interictal )
        print('Preictal samples:', number_preictal )
        
        X = get_X_from_files(base_dir_train, data_files_all['file'])
        y = np.repeat(data_files_all['class'], 16, axis=0)
        
        pickle.dump(X, open('X_' + FEATURES_NAME + '_patient_' + str(PATIENT_NUMBER) +'.pkl', 'wb'))
        pickle.dump(y, open('y_' + FEATURES_NAME + '_patient_' + str(PATIENT_NUMBER) +'.pkl', 'wb'))
        
timera = time.clock()    
data_pickle()
timerb = time.clock()
print('Loading complete. CPU Wall time for loading, in seconds : ', timerb-timera)