# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 17:51:29 2016

@author: Pierre H. Richemond
"""
import scipy.io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.signal import correlate, resample
import time
import h5py
from functools import partial
from multiprocessing import Pool

from sklearn.preprocessing import normalize, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve

import pickle
from sklearn.externals import joblib

rnd_seed = 1337
np.random.seed(rnd_seed)
base_dir_train = u'train_all/'
base_dir_tests = u'test_all/'
INTERICTAL = 0
PREICTAL = 1

def get_class_from_name(name):
    """
    Gets the class from the file name.
    The class is defined by the last number written in the file name, e.g.
    Input: ".../1_1_1.mat"
    Output: 1.0
    Input: ".../1_1_0.mat"
    Output: 0.0
    """
    try:
        return float(name[-5])
    except:
        return 0.0

assert get_class_from_name('/train_1/1_1_0.mat') == 0.0
assert get_class_from_name('/train_1/1_1_1.mat') == 1.0

def get_file_names_and_classes(base_dir):
    ignored_files = ['.DS_Store', '1_45_1.mat']
    
    return np.array(
        [
            (file, get_class_from_name(file)) 
            for file in os.listdir(base_dir) if file not in ignored_files
        ],
        dtype=[('file', '|S16'), ('class', 'float32')]
    )

    
data_files_all = get_file_names_and_classes(base_dir_train)
# Count the occurrences of Interictal and Preictal classes
unique, counts = np.unique(data_files_all['class'], return_counts=True)
occurrences = dict(zip(unique, counts))

number_interictal = occurrences.get(INTERICTAL)
number_preictal = occurrences.get(PREICTAL)
print('Interictal samples:', number_interictal )
print('Preictal samples:', number_preictal )

# Multiple that tells us how many interictal cases to load for each preictal
# Generally for logistic regression 5 times more controls than cases is OK
inter_pre_ratio = float(5.0)
set_size_interictal = np.floor(inter_pre_ratio * number_preictal)
set_size_preictal = number_preictal

# Randomly select an equal-size set of Interictal and Preictal samples
data_random_interictal = np.random.choice(data_files_all[data_files_all['class'] == 0], size=set_size_interictal)
data_random_preictal = np.random.choice(data_files_all[data_files_all['class'] == 1], size=set_size_preictal)
# Merge the data sets and shufle the collection
data_files = np.concatenate([data_random_interictal, data_random_preictal])
data_files.dtype = data_files_all.dtype  # Sets the same dtype than the original collection
np.random.shuffle(data_files)

print(data_files.shape, data_files.size)

def get_X_from_files(base_dir, files, show_progress=True):
    """
    Given a list of filenames, returns the final data we want to train the models.
    """
    X = None
    total_files = len(files)

    for i, filename in enumerate(files):
        if show_progress and i % int(total_files / 10) == 0:
            print(u'%{}: Loading file {}'.format(int(i * 100 / total_files), filename))

        try:
            mat_data = scipy.io.loadmat(''.join([base_dir, filename.decode('UTF-8')]))
        except ValueError as ex:
            print(u'Error loading MAT file {}: {}'.format(filename, str(ex)))
            continue

        # Gets a 16x240000 matrix => 16 channels reading data for 10 minutes at 400Hz
        channels_data = mat_data['dataStruct'][0][0][0].transpose()
        # Resample each channel to get only a meassurement per second
        # 10 minutes of measurements, grouping data on each second
        channels_data = resample(channels_data, 600, axis=1, window=400)
        
        # It seems that adding bivariate meassurements helps a lot on
        # signals pattern recognition.
        # For each channel, add the correlation meassurements with all the other
        # channels.
        # TODO: This should be done in a more efficient way ¯\_(ツ)_/¯
        correlations = None
        for i in range(16):
            correlations_i = np.array([])
            for j in range (16):
                if i != j:
                    corr_i = correlate(channels_data[i], channels_data[j], mode='same')
                    correlations_i = np.concatenate([correlations_i, corr_i])
                    
            if correlations is None:
                correlations = correlations_i
            else:
                correlations = np.vstack([correlations, correlations_i])

        channels_data = np.column_stack([channels_data, correlations])
        
        X = np.vstack([X, channels_data]) if X is not None else channels_data
    
    return X

    
X = get_X_from_files(base_dir_train, data_files['file'])
y = np.repeat(data_files['class'], 16, axis=0)

print('X_shape:', X.shape, 'X_size:', X.size)
print('y_shape:', y.shape, 'y_size:', y.size)

# Normalizes the data
normalize(X, copy=False)

# Plots a user normalized sample
matplotlib.rcParams['figure.figsize'] = (20.0, 5.0)
print('Showing case of file:', data_files['file'][0])
for i in range(16):
    plt.subplot(8, 2, i + 1)
    plt.plot(X[i])
    
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=rnd_seed)
clf = linear_model.LogisticRegression(C=16, n_jobs=1, solver='liblinear', verbose=5)

# Fit the logistic regression model
timera = time.clock()
clf.fit(X_train, y_train)
timerb = time.clock()
print('CPU Wall time for fit, in seconds : ', timerb-timera)

# Serialize / pickle save model
model_filename = 'final_logistic_regression.sav'
pickle.dump(clf, open(model_filename, 'wb'))

y_pred = clf.predict(X_test)
y_decision = clf.predict_proba(X_test)

print(u'Accuracy:', accuracy_score(y_test, y_pred))
print(u'Precision:', precision_score(y_test, y_pred))
print(u'Recall:', recall_score(y_test, y_pred))
print(u'F1 score:', f1_score(y_test, y_pred, average='binary'))
print(u'ROC AUC:', roc_auc_score(y_test, y_decision))