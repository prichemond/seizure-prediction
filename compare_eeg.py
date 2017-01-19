# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 03:01:49 2016

@author: Pierre H. Richemond
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.io import loadmat

class eeg(object):
    def __init__(self, filename):
        self.preictal = None
        self.data = None
        self.samplingrate = None
        self.patientid = None
        self.runnum = None
        self.readMat(filename)

    def readMat(self, filename):
        m = loadmat(filename)
        self.data = pd.DataFrame(m['dataStruct']['data'][0, 0])
        self.data.columns = [int(i) for i in m['dataStruct']['channelIndices'][0, 0][0]]
        self.samplingrate = m['dataStruct']["iEEGsamplingRate"][0, 0][0, 0]
        #print(self.samplingrate)
        self.data["seconds"] = pd.to_numeric(self.data.index) / self.samplingrate
        tmp = os.path.basename(filename)[:-4].split("_")
        self.patientid = int(tmp[0])
        self.runnum = int(tmp[1])
        if (len(tmp) == 3):
            self.preictal = int(tmp[2])

    def stripDropOut(self, winlen=11):
        dropout = None
        count = 0
        for ch in [1, 3, 7, 11, 14]:
            tmp = self.data[self.data[ch].rolling(window=winlen, center=True).std() < 0.00001][["seconds", ch]]
            if (tmp.shape[0] > count):
                count = tmp.shape[0]
                dropout = tmp
        if (count > 0):
            self.data = self.data.drop(dropout.index)
            starttime = self.data.head(1)["seconds"].values[0]
            self.data.index = range(self.data.shape[0])
            self.data["seconds"] = self.data.index / self.samplingrate
            self.data["seconds"] = self.data["seconds"] + starttime

e0 = eeg("./Data/train_1/1_100_0.mat")
e0.stripDropOut()
print(e0.data.shape, e0.patientid, e0.runnum, e0.preictal)

e1 = eeg("./Data/train_1/1_100_1.mat")
e1.stripDropOut()
# offset start of second EEG
e1.data["seconds"] = e1.data["seconds"] + e0.data.tail(1)["seconds"].values[0]
print(e1.data.shape, e1.patientid, e1.runnum, e1.preictal)

for ch in e0.data.columns:
    if (ch != "seconds"):
        f = plt.figure(figsize=(12,2))
        plt.plot(e0.data["seconds"], e0.data[ch])
        plt.plot(e1.data["seconds"], e1.data[ch], color="red")
        plt.title("Channel %d" % ch)
        
r = 0
c = 0
fig, ax = plt.subplots(4, 4, figsize=(12,12))
for ch in e0.data.columns:
    if (ch != "seconds"):
        sns.distplot(e0.data[ch], ax=ax[r][c], kde=False, norm_hist=True)
        sns.distplot(e1.data[ch], ax=ax[r][c], color="red", kde=False, norm_hist=True)
        c += 1
        if (c >=4):
            r += 1
            c = 0 
            