# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 20:39:08 2016

@author: Pierre H. Richemond
"""
from __future__ import print_function
import numpy as np
import pickle

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc

from keras.layers.advanced_activations import ELU
from keras.utils import np_utils
from keras import backend as K

def data():
    
    PATIENT_NUMBER = 1
    VAL_HOLDOUT = 0.20

    TOTAL_PATIENTS = 3
    NB_CLASSES = 2
    RND_SEED = 1337
    
    features_name = 'X_FFT_patient_' + str(PATIENT_NUMBER) + '.pkl'
    features_array = pickle.load(open(features_name,'rb'))

    y_name = 'y_FFT_patient_' + str(PATIENT_NUMBER) + '.pkl'
    y_array = pickle.load(open(y_name,'rb'))
    
    # Get rid of -inf due to log of zero spectrum
    features_array[np.isneginf(features_array) ] = -8888.8

    shape_vec = features_array.shape
    features_array = features_array.reshape( 1, shape_vec[0], shape_vec[1], shape_vec[2])
    features_array = np.transpose(features_array, (3,0,1,2) )
    features_array = features_array.astype('float32')
    
    nlen = int(y_array.shape[0] / 16)
    ynew = np.zeros(nlen).astype('uint8')
    for i in range(nlen):     
        ynew[i] = y_array[16*i]

    bigynew = np_utils.to_categorical(ynew, NB_CLASSES)
    
    X_train, X_test, Y_train, Y_test = train_test_split(features_array, bigynew, test_size= VAL_HOLDOUT, random_state=RND_SEED )
    return X_train, X_test, Y_train, Y_test
    
# Now for the Keras model definition.
def model():
    
    NFILTER1 = {{choice([16, 24, 32])}}
    NFILTER2 = {{choice([32, 48, 64])}}
    NFILTER3 = {{choice([64, 128, 256])}}
    BATCH_NORM = False
    
    model = Sequential()

    model.add(Convolution2D(NFILTER1, 96, 1, input_shape=(1,96,10), border_mode='valid', init='glorot_uniform' ))
    if BATCH_NORM == True:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Reshape((1,NFILTER1,10)) )
    model.add(Convolution2D(NFILTER2, NFILTER1, 2, border_mode='valid', init='glorot_uniform'))
    if BATCH_NORM == True:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dropout({{uniform(0,1)}}))
    model.add(Dense(NFILTER3, W_regularizer=l2(0.0001)))
    if BATCH_NORM == True:
        model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout({{uniform(0,1)}}))
    model.add(Dense(2, W_regularizer=l2(0.0001) ))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
              metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    checkpointer = ModelCheckpoint(filepath='keras_weights.hdf5',
                                   verbose=1,
                                   save_best_only=True)
    
    model.fit(X_train, Y_train,
              batch_size={{choice([1, 10, 64])}},
              nb_epoch=20,
              verbose=2,
              validation_data=(X_test, Y_test),
            callbacks=[early_stopping, checkpointer])
    
    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)
    #print('Test score:', score)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    
    X_train, Y_train, X_test, Y_test = data()

    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=100,
                                          trials=Trials())
    
    #ensemble_model = optim.best_ensemble(nb_ensemble_models=5,
    #                                     model=model, data=data,
    #                                     algo=rand.suggest, max_evals=10,
    #                                     trials=Trials(),
    #                                     voting='hard')
    
    
    best_model.save('my_best_CNN.h5')
    print(best_model.to_yaml())