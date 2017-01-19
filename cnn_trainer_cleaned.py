# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 20:39:08 2016

@author: Pierre H. Richemond
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, l1
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import train_test_split, KFold, StratifiedShuffleSplit, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from keras.utils import np_utils

PATIENT_NUMBER = 1

TOTAL_PATIENTS = 3
NB_CLASSES = 2
INTERICTAL = 0
PREICTAL = 1
REBALANCE_CLASSES_EVENLY = True

FREQ_BANDS = 6
N_CHANNELS = 16
RND_SEED = 1337

def scale_across_time(x, x_test=None, scalers=None):
    n_examples = x.shape[0]
    n_channels = x.shape[1]
    n_fbins = x.shape[2]
    n_timesteps = x.shape[3]

    if scalers is None:
        scalers = [None] * n_channels

    for i in range(n_channels):
        xi = np.transpose(x[:, i, :, :], axes=(0, 2, 1))
        xi = xi.reshape((n_examples * n_timesteps, n_fbins))

        if x_test is not None:
            xi_test = np.transpose(x_test[:, i, :, :], axes=(0, 2, 1))
            xi_test = xi_test.reshape((x_test.shape[0] * n_timesteps, n_fbins))
            xi_complete = np.vstack((xi, xi_test))
        else:
            xi_complete = xi

        if scalers[i] is None:
            scalers[i] = StandardScaler()
            scalers[i].fit(xi_complete)

        xi = scalers[i].transform(xi)

        xi = xi.reshape((n_examples, n_timesteps, n_fbins))
        xi = np.transpose(xi, axes=(0, 2, 1))
        x[:, i, :, :] = xi
    return x, scalers

def data_renormalizer(x):    
    x[np.isneginf(x) ] = -8888.8

    x = np.transpose(np.asarray(np.vsplit(x, 16)), (3,0,1,2) )
    x, scale = scale_across_time(x)
    x = np.vstack(np.transpose(x, (1,2,3,0) ))
    return x  
    
def data():
    basestring = '_FFT_patient_' + str(PATIENT_NUMBER) + '.pkl'
    features_array = pickle.load(open('X' + basestring,'rb'))
    y_array = pickle.load(open('y' + basestring,'rb'))
    
    features_array = data_renormalizer(features_array)
    
    shape_vector = features_array.shape
    ncols = shape_vector[0]
    ntime = shape_vector[1]
    nsamples = shape_vector[2]
    nlen = int(y_array.shape[0] / 16)
    assert(nlen==nsamples)
    
    ynew = np.zeros(nlen)
    for i in range(nlen):
        ynew[i] = y_array[16*i]

    features_array = features_array.reshape( 1, ncols, ntime, nsamples)
    features_array = np.transpose(features_array, (3,0,1,2) )
    features_array = features_array.astype('float32')
    
    if REBALANCE_CLASSES_EVENLY == True:
        features_array_preictal = features_array[ynew==1,:,:,:]
        features_array_interictal = features_array[ynew==0,:,:,:]
        npreictal = features_array_preictal.shape[0]
        ninterictal = features_array_interictal.shape[0]
        copied_preictal = np.repeat(features_array_preictal, ninterictal // npreictal, axis=0)
        features_array = np.concatenate( (features_array_interictal, copied_preictal) )
        ynew = np.concatenate( (np.zeros(ninterictal),np.ones(copied_preictal.shape[0])) )
        ynew = ynew.astype('float32')
    
    Ynew = np_utils.to_categorical(ynew, NB_CLASSES)    
    return features_array, ynew, Ynew

def create_model(optimizer='adadelta',init='he_normal', nfilter1 = 10, nfilter2 = 32, nfilter3 = 1024):
    
    NFILTER1 = nfilter1
    NFILTER2 = nfilter2
    NFILTER3 = nfilter3
    BATCH_NORM = True
    model = Sequential()

    model.add(Convolution2D(NFILTER1, 96, 1, input_shape=(1,96,10), border_mode='valid', init=init, W_regularizer=l2(0.0001) ))
    
    if BATCH_NORM == True:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Flatten())
    model.add(Reshape((1,NFILTER1,10)) )
    model.add(Convolution2D(NFILTER2, NFILTER1, 2, border_mode='valid', init=init))
    
    if BATCH_NORM == True:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Flatten())
    
    model.add(Dense(NFILTER3, W_regularizer=l2(0.0001), init=init ))
    
    if BATCH_NORM == True:
        model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(0.95))
    
    #model.add(Dense(16))   # optional
    #model.add(Activation('tanh')) # optional
    #model.add(Dropout(0.5)) # optional
    
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    #print(model.output_shape)
    return model

# Main    
np.random.seed(RND_SEED)

BATCH_SIZE = 300
NB_EPOCH = 500
STRATIFIED_SPLIT = True
GRID_SEARCH = False

model = create_model()

early_stopping = EarlyStopping(monitor='val_loss', patience=100)
checkpointer = ModelCheckpoint(filepath='keras_weights_cnn.hdf5',
                                   verbose=1,
                                   save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                  patience=50, min_lr=0.001)
sklearn_model = KerasClassifier(build_fn = create_model, verbose=0 )

optimizers = ['adadelta']
init = ['normal']
epochs = np.array([100])
batches = np.array([300, 400, 500, 600, 700, 800, 900, 1000])
nfilter1= np.array([16])
nfilter3= np.array([128])
p_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches, init=init, nfilter1=nfilter1, nfilter3=nfilter3 )

X, y, Y = data()
if STRATIFIED_SPLIT == True:
    sss = StratifiedShuffleSplit(y, n_iter=1, test_size=0.25, random_state=RND_SEED)
    for train_idx, test_idx in sss:
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
    Y_train, Y_test = np_utils.to_categorical(y_train, NB_CLASSES), np_utils.to_categorical(y_test, NB_CLASSES)
else:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.25, random_state=RND_SEED )

grid = GridSearchCV(sklearn_model, p_grid, cv=sss)

if GRID_SEARCH == True :
    grid_result = grid.fit(X, Y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    for params, mean_score, scores in grid_result.grid_scores_:
        print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
else:    
    model.fit(X_train, Y_train, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH,
          verbose=1, validation_data=(X_test, Y_test), callbacks=[checkpointer, reduce_lr, early_stopping ])
    score = model.evaluate(X_test, Y_test, verbose=1)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    y_predict = model.predict(X_train)
    y_delta = Y_train - y_predict
    auc_convnet = roc_auc_score(Y_train[:,0], y_predict[:,0])
    print('AUC estimation:', auc_convnet)
#print(classification_report(y_train, y_predict[:,0]))
#Let's visualize learned filters
#layer_dict = dict([(layer.name, layer) for layer in model.layers])
    for lay in model.layers:
        W = lay.W.get_value(borrow=True)
        W = np.squeeze(W)
        print(W.shape)
        plt.figure(figsize=(20,10))            
        plt.imshow(W, cmap='hot',interpolation='nearest')
        plt.show()
        input("Press Enter to continue...")