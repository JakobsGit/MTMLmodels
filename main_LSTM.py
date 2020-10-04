# -*- coding: utf-8 -*-

#import used libraries
import numpy as np
import math
import pandas as pd

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

import skopt
from skopt import gp_minimize 
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from skopt import dump, load

import preprocessing_data_helpers
import get_data_functions

from get_data_functions import *
from preprocessing_data_helpers import *

#set random seed
randomState = 1
np.random.seed(randomState)
try:
    tf.random.set_seed(randomState)
except:
    tf.set_random_seed(randomState)

from tensorflow.keras.metrics import *


#####################################################
# Input / Parameter Definition
#####################################################

# select approach and timesteps
# approach 31: RF approach (classification)
#----------------
# Feature vector: [R_{t,1}, R_{t,2},..., R_{t,20}, R_{t,40}, R_{t,60},..., R_{t,240}] 
# where  R_{t,m} = P_t / P_{t-m} - 1 with R_{t,m}: Return calculated with P_t closing prices 
#
# Target: if stock return > median return of set of stock -> 1 else: 0
#
# approach 240: LSTM approach (classification)
#----------------
# Feature vector: [R_{t-239}, R_{t-238},..., R_{t}] 
# where R_t = P_t / P_{t-1} -1 with  R_{t,m}: Return calculated with P_t closing prices 
# The features are standardized (std dev = 1, mean = 0) with the training data
#
# Target: if stock return > median return of set of stock -> 1 else: 0
#
# approach 63: LSTM approach (regression)
#----------------
#Feture vector: [ R_t-62_n, R_t-61_n, ...R_t_n]
# where R_t_n = P_t - P_{t-n}  with R_{t,m}: Return calculated with P_t closing prices 
#
#Target: R_t+n_n 

# timesteps have to be adjusted for approach 63 to e.g. 63


approach = 240
timesteps = 240

# define parameter space
if approach == 240:
    dim_learning_rate = Real(low=1e-4, high=1e-1, prior='log-uniform', name='learning_rate')
    dim_num_hidden_layers = Integer(low=1, high=2, name='num_hidden_layers')
    dim_num_lstm_nodes = Integer(low=25, high=100, name='num_lstm_nodes')
    dim_num_batch_size = Integer(low=5, high=9, name='num_batch') # batch_size= 2^num_batch
    dim_dropout_rate = Real(low=0.1, high=0.5, prior='log-uniform', name='dropout_rate')

elif approach == 63:
    dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform', name='learning_rate')
    dim_num_hidden_layers = Integer(low=1, high=2, name='num_hidden_layers')
    dim_num_lstm_nodes = Integer(low=50, high=300, name='num_lstm_nodes')
    dim_num_batch_size = Integer(low=0, high=5, name='num_batch') # batch_size= 2^num_batch
    dim_dropout_rate = Real(low=0.1, high=0.5, prior='log-uniform', name='dropout_rate')


dimensions = [dim_learning_rate,
              dim_num_hidden_layers,
              dim_num_lstm_nodes,
              dim_num_batch_size,
              dim_dropout_rate]

# define default parameters to start the hyperparameter optimization
lr = 0.01
layers = 1
nodes = 50
batch = 5
dropout = 0.1

default_parameters = [lr, layers, nodes, batch, dropout]

number_of_folds = 15

# list of delta_t that indicates target return after delta_t trading days
nlist = [20,10,5,3,2,1]

#define parameter & loss results dataframe  
zeroph = np.zeros(len(nlist))
parameter_dict = {'logloss':zeroph, 'lr':zeroph,'layers':zeroph, 'node':zeroph, 'batch':zeroph, 'dropout':zeroph}
parameter_df = pd.DataFrame(parameter_dict, columns = ['logloss','lr','layers', 'node','batch', 'dropout'], index = nlist) 

# get s&p 500 data with highest trade volume from yahoo finance 
stockdata = getsp500data(numberofstocks=50,startdate='2004-12-31', enddate='2019-12-31')  
stockfullhistcheck(stockdata)
replacenans(stockdata)


def create_lstm_model(timesteps, learning_rate,num_hidden_layers,
                 num_lstm_nodes,dropout_rate,approach):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_lstm_nodes:   Number of lstm nodes
    dropout_rate:         Drop-out value
    num_hidden_layers: Number of hidden layers
    
    """
    
    model = Sequential()
    if approach == 63:
         #model.add(Bidirectional(LSTM(units = num_lstm_nodes, return_sequences = False, input_shape = (timesteps, 1))))
        if num_hidden_layers == 1:
          model.add(Bidirectional(LSTM(units = num_lstm_nodes, return_sequences = False, input_shape = (timesteps, 1))))
        elif num_hidden_layers == 2:
          model.add(Bidirectional(LSTM(units = num_lstm_nodes, return_sequences = True, input_shape = (timesteps, 1))))
          model.add(Dropout(dropout_rate))
          model.add(Bidirectional(LSTM(units = num_lstm_nodes, return_sequences = False)))
        elif num_hidden_layers == 3:
          model.add(Bidirectional(LSTM(units = num_lstm_nodes, return_sequences = True, input_shape = (timesteps, 1))))
          model.add(Dropout(dropout_rate))
          model.add(Bidirectional(LSTM(units = num_lstm_nodes, return_sequences = True)))
          model.add(Dropout(dropout_rate))
          model.add(Bidirectional(LSTM(units = num_lstm_nodes, return_sequences = False)))

    else:
        #model.add(LSTM(units = num_lstm_nodes, return_sequences = False, input_shape = (timesteps, 1)))
        if num_hidden_layers == 1:
          model.add(LSTM(units = num_lstm_nodes, return_sequences = False, input_shape = (timesteps, 1)))
        elif num_hidden_layers == 2:
          model.add(LSTM(units = num_lstm_nodes, return_sequences = True, input_shape = (timesteps, 1)))
          model.add(Dropout(dropout_rate))
          model.add(LSTM(units = num_lstm_nodes, return_sequences = False))
        elif num_hidden_layers == 3:
          model.add(LSTM(units = num_lstm_nodes, return_sequences = True, input_shape = (timesteps, 1)))
          model.add(Dropout(dropout_rate))
          model.add(LSTM(units = num_lstm_nodes, return_sequences = True))
          model.add(Dropout(dropout_rate))
          model.add(LSTM(units = num_lstm_nodes, return_sequences = False))
    
    model.add(Dropout(dropout_rate))

    model.add(Dense(units = 1))   

    if approach == 240:
        model.add(Activation('sigmoid'))
        model.compile(optimizer = RMSprop(lr=learning_rate, clipvalue=0.5),
            loss = 'binary_crossentropy',
            metrics=['acc','binary_crossentropy'])
    
    elif approach == 63:
        model.add(Activation('linear'))
        model.compile(optimizer = Adam(lr=learning_rate, clipvalue=0.5),
            loss = 'mean_squared_error',
            metrics=['mean_squared_error'])
    
    return model  


# run time series validation, returing the average validation loss (out of sample validation)
def timeseriesCV(dataset, X,y,y_df,fold_size, numberofdays, timesteps, learning_rate,num_hidden_layers, num_lstm_nodes, num_batch, dropout_rate, number_of_folds, approach):
  val_loss_list = []
  if approach == 63:
    foldlimit = int(number_of_folds/3)
  else:
    foldlimit = int(number_of_folds*0.6)

  for foldindex in range(1,foldlimit):

    X_train, X_val, X_test, y_train, y_val, y_test, y_train_df, y_val_df, y_test_df  = createfolds(dataset, X, y, y_df, fold_size, foldindex, numberofdays, timesteps, approach)
    

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
    #X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    model = create_lstm_model(timesteps=timesteps,
                        learning_rate = learning_rate,
                        num_hidden_layers = num_hidden_layers,
                        num_lstm_nodes= num_lstm_nodes,
                        dropout_rate = dropout_rate,
                        approach = approach)
    
    if approach == 240:
        es = EarlyStopping(monitor='val_binary_crossentropy', 
                           mode='min', 
                           verbose=0, 
                           patience=10)
        
        checkpointer = ModelCheckpoint(filepath="weights.hdf5",
                                       monitor='val_binary_crossentropy',
                                       mode ='min',
                                       verbose=0, 
                                       save_best_only=True)
        
    elif approach == 63:
        es = EarlyStopping(monitor='val_mean_squared_error', 
                           mode='min', 
                           verbose=1, 
                           patience=10)
        
        checkpointer = ModelCheckpoint(filepath="weights.hdf5",
                                       monitor='val_mean_squared_error',
                                       mode ='min',
                                       verbose=0, 
                                       save_best_only=True)

    # Use Keras to train the model.
    history = model.fit(x=X_train,
                        y=y_train,
                        epochs=2,
                        batch_size=2**num_batch,
                        validation_data=(X_val,y_val),
                        verbose=0,
                        callbacks=[es, checkpointer])

    
    val_loss_hist = history.history['val_loss']
    val_loss_hist = [x for x in val_loss_hist if (not math.isnan(x))]
    val_loss = np.min(val_loss_hist)

    val_loss_list.append(val_loss)

    del model

    K.clear_session()
    
  val_loss_list = [x for x in val_loss_list if (not math.isnan(x))]

  av_val_loss = np.average(val_loss_list)
  
  return av_val_loss

# function to save the best parameter set for each delta_t
def saveoptresults(search_result, parameter_df, forecastdays, approach):

  learning_rate = search_result.x[0]
  num_hidden_layers = search_result.x[1]
  num_lstm_nodes = search_result.x[2]
  num_batch = search_result.x[3]
  dropout_rate = search_result.x[4]
  
  parameter_df.loc[forecastdays,'logloss'] = sorted(zip(search_result.func_vals, search_result.x_iters))[0][0]

  parameter_df.loc[forecastdays,'lr'] = learning_rate
  parameter_df.loc[forecastdays,'node'] = num_lstm_nodes
  parameter_df.loc[forecastdays,'layers'] = num_hidden_layers
  parameter_df.loc[forecastdays,'batch'] = 2**num_batch
  parameter_df.loc[forecastdays,'dropout'] = dropout_rate

  parameterdfdir = 'parameter_df_LSTM_appr' + str(approach) +".csv"
  with open(parameterdfdir, 'w') as csv_file:
      parameter_df.to_csv(path_or_buf=csv_file,  index=False)

  optimizationdirect = 'optresult_LSTM_appr' + str(approach)  +'n'+ str(forecastdays) +'.pkl'
  dump(search_result, optimizationdirect)
  return parameter_df

# train model with best hyperparameters and return the predictions for the test data (out of sample)
def savetestresults(dataset, X, y, y_df,approach, fold_size, number_of_folds, forecastdays, numberofdays, timesteps, search_result):    

  learning_rate = search_result.x[0]
  num_hidden_layers = search_result.x[1]
  num_lstm_nodes = search_result.x[2]
  num_batch = search_result.x[3]
  dropout_rate = search_result.x[4]

  if approach == 63:
    firstfold = int(number_of_folds/3)
    foldlimit = firstfold+3
  else:
    firstfold = int(number_of_folds*0.6)
    foldlimit = firstfold+3

  testfolds = range(firstfold,foldlimit)#range(int(number_of_folds*0.8-2),number_of_folds-2)

  for foldindex in testfolds:
    X_train, X_val, X_test, y_train, y_val, y_test, y_train_df, y_val_df, y_test_df = createfolds(dataset, X, y, y_df, fold_size, foldindex, numberofdays, timesteps, approach)
    print('X', X_test.shape)
    print('y', y_test.shape)
    print('df', y_test_df.shape)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = create_lstm_model(timesteps=timesteps,
                        learning_rate = learning_rate,
                        num_hidden_layers = num_hidden_layers,
                        num_lstm_nodes= num_lstm_nodes,
                        dropout_rate = dropout_rate,
                        approach = approach)
    
      
    if approach == 240:
        es = EarlyStopping(monitor='val_binary_crossentropy', 
                            mode='min', 
                            verbose=0, 
                            patience=10)
        
        checkpointer = ModelCheckpoint(filepath="weights.hdf5",
                                        monitor='val_binary_crossentropy',
                                        mode ='min',
                                        verbose=0, 
                                        save_best_only=True)
        
    elif approach == 63:
        es = EarlyStopping(monitor='val_mean_squared_error', 
                            mode='min', 
                            verbose=1, 
                            patience=20)
        
        checkpointer = ModelCheckpoint(filepath="weights.hdf5",
                                        monitor='val_mean_squared_error',
                                        mode ='min',
                                        verbose=1, 
                                        save_best_only=True)

    # Use Keras to train the model.
    history = model.fit(x=X_train,
                        y=y_train,
                        epochs=2,#num_epochs,
                        batch_size=2**num_batch,
                        validation_data=(X_val,y_val),
                        verbose=0,
                        callbacks=[es, checkpointer])

    print("Test Loss", np.min(history.history['val_loss']))
    model.load_weights('weights.hdf5')

    y_test_df['forecastdays'] = forecastdays
    y_test_df['fold'] = foldindex+4

    y_val_pred = model.predict(X_test)
    y_test_df['Prediction'] = y_val_pred

    if foldindex == testfolds[0]:
      test_df = y_test_df.copy()
    else:
      test_df = pd.concat([test_df, y_test_df])
    
    del model
    
    K.clear_session()

  directory = 'test_df_LSTM_n_' +str(forecastdays) + 'appr'+ str(approach) + '.csv'
  with open(directory, 'w') as csv_file:
      test_df.to_csv(path_or_buf=csv_file,  index=False)
      
  return test_df

# optimization function for the hyperparameter optimization
@use_named_args(dimensions=dimensions)
def fitness(learning_rate,
            num_hidden_layers,
            num_lstm_nodes,
            num_batch,
            dropout_rate):

    # Print the hyper-parameters.
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('num hidden layers:', num_hidden_layers)
    print('num lstm nodes:', num_lstm_nodes)
    print('batch size:', 2**num_batch)
    print('dropout:', dropout_rate)
    print()
    
    # Create the neural network with these hyper-parameters + run time series validation 
    
    av_val_loss = timeseriesCV(dataset, X,y,y_df,fold_size, numberofdays, timesteps, learning_rate,num_hidden_layers, num_lstm_nodes, num_batch, dropout_rate, number_of_folds, approach)
    
    print()
    print("Average Loss: ", (av_val_loss))
    print()
    # NOTE: Scikit-optimize does minimization so it tries to
    # find a set of hyper-parameters with the LOWEST fitness-value.
    # Because we are interested in the HIGHEST classification
    # accuracy, we need to negate this number so it can be minimized.
    return av_val_loss
    

##########################################################
# HYPERPARAMETER OPTIMIZATION
##########################################################

for forecastdays in nlist:
  print("performance forecast for ", forecastdays, "days")
  print(" ")
  
  #data preprocessing
  dataset = createreturncolumn(stockdata,forecastdays,approach)
  dataset = createtargetcolumn(dataset,approach)
  dataset = deletedividendentries(dataset)

  X, y, y_df = createseries(dataset, timesteps=timesteps, approach=approach, n=forecastdays)
  
  numberofdays = np.unique(y_df.Date).shape[0]
  fold_size = int(numberofdays/number_of_folds)
  
  
  # bayesian optimization
  search_result = gp_minimize(func=fitness,
                              dimensions=dimensions,
                              acq_func='EI', # Expected Improvement.
                              n_calls=11,
                              x0=default_parameters)
  
  
  parameter_df = saveoptresults(search_result, parameter_df, forecastdays, approach)
  
  test_df = savetestresults(dataset, X, y, y_df,approach, fold_size, number_of_folds, forecastdays, numberofdays, timesteps, search_result)

