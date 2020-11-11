# -*- coding: utf-8 -*-

#import used libraries
import numpy as np
import math
import pandas as pd

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
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
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
from sklearn.metrics import  mean_squared_error, mean_absolute_error
#####################################################
# Input / Parameter Definition
#####################################################

# select approach, timesteps and arch
# arch: 'LSTM' or 'GRU'
# approach 31: RF approach (classification)
#----------------
# Feature vector: [R_{t,1}, R_{t,2},..., R_{t,20}, R_{t,40}, R_{t,60},..., R_{t,240}], Open, Close, High and Low Prices
# where  R_{t,m} = P_t / P_{t-m} - 1 with R_{t,m}: Return calculated with P_t closing prices 
#
# Target: if stock return > median return of set of stocks -> 1 else: 0
#
# approach 240: LSTM approach (classification)
#----------------
# Feature vector: [R_{t-239}, R_{t-238},..., R_{t}], Open, Close, High and Low Prices
# where R_t = P_t / P_{t-1} -1 with  R_{t,m}: Return calculated with P_t closing prices 
# The features are standardized (std dev = 1, mean = 0) with the training data
#
# Target: if stock return >0 -> 1 else: 0
#
# approach 241: LSTM approach (classification)
#----------------
# Feature vector: [R_{t-239}, R_{t-238},..., R_{t}], Open, Close, High and Low Prices
# where R_t = P_t / P_{t-1} -1 with  R_{t,m}: Return calculated with P_t closing prices 
# The features are standardized (std dev = 1, mean = 0) with the training data
#
# Target: R_t+n_n
# where R_t_n = P_t - P_{t-n}  with R_{t,m}: Return calculated with P_t closing prices 
#

arch = 'LSTM'
approach = 241
timesteps = 20

# define parameter space
if (approach == 240) | (approach == 241):
    dim_learning_rate = Real(low=1e-4, high=1e-1, prior='log-uniform', name='learning_rate')
    dim_num_hidden_layers = Integer(low=1, high=2, name='num_hidden_layers')
    dim_num_lstm_nodes = Integer(low=25, high=100, name='num_lstm_nodes')
    dim_num_batch_size = Integer(low=7, high=12, name='num_batch') # batch_size= 2^num_batch
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
batch = 12
dropout = 0.1

default_parameters = [lr, layers, nodes, batch, dropout]

number_of_folds = 15

best_test_AUC = 0.0
best_test_RMSE = 99999

# list of delta_t that indicates prediction of target return after delta_t trading days
nlist = [20,10,5,3,2,1]

#define parameter & loss results dataframe  
zeroph = np.zeros(len(nlist))
parameter_dict = {'auc':zeroph, 'lr':zeroph,'layers':zeroph, 'node':zeroph, 'batch':zeroph, 'dropout':zeroph}
parameter_df = pd.DataFrame(parameter_dict, columns = ['auc','lr','layers', 'node','batch', 'dropout'], index = nlist) 

# define dataframe to save the performace on the test set during the hyperparameter optimization 
test_perf_dict = {'auc':np.zeros(1), 'acc':np.zeros(1),'balacc':np.zeros(1), 'RMSE':np.zeros(1), 'MSE':np.zeros(1),'MAE':np.zeros(1)}
test_perf_df = pd.DataFrame(test_perf_dict, columns = ['auc','acc','balacc','RMSE','MSE','MAE']) 


# get s&p 500 data with highest trade volume from yahoo finance and replace the "nan"
stockdata = getsp500data(numberofstocks=50,startdate='2004-12-31', enddate='2019-12-31')  
#stockfullhistcheck(stockdata)
replacenans(stockdata)

def create_lstm_model(timesteps, learning_rate,num_hidden_layers,
                 num_lstm_nodes,dropout_rate,approach):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    dim_num_hidden_layers: Number of hidden layers
    num_lstm_nodes:   Number of lstm nodes
    dropout_rate:         Drop-out value
    num_hidden_layers: Number of hidden layers
    
    """
    model = Sequential()
    if approach == 63:
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
        if num_hidden_layers == 1:
          model.add(LSTM(units = num_lstm_nodes, return_sequences = False, input_shape = (timesteps, 5)))
        elif num_hidden_layers == 2:
          model.add(LSTM(units = num_lstm_nodes, return_sequences = True, input_shape = (timesteps, 5)))
          model.add(Dropout(dropout_rate))
          model.add(LSTM(units = num_lstm_nodes, return_sequences = False))
        elif num_hidden_layers == 3:
          model.add(LSTM(units = num_lstm_nodes, return_sequences = True, input_shape = (timesteps, 5)))
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
            metrics=['acc','binary_crossentropy', tf.keras.metrics.AUC(name='auc')])
    
    elif (approach == 63) | (approach == 241):
        model.add(Activation('linear'))
        model.compile(optimizer = Adam(lr=learning_rate, clipvalue=0.5),
            loss = 'mean_squared_error',
            metrics=[tf.keras.metrics.RootMeanSquaredError(name='RMSE'),
                     'mean_squared_error',
                     tf.keras.metrics.MeanAbsoluteError(name='MAE', dtype=None)])
    return model  


def create_GRU_model(timesteps, learning_rate,num_hidden_layers,
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
        if num_hidden_layers == 1:
          model.add(GRU(units = num_lstm_nodes, return_sequences = False, input_shape = (timesteps, 5)))
        elif num_hidden_layers == 2:
          model.add(GRU(units = num_lstm_nodes, return_sequences = True, input_shape = (timesteps, 5)))
          model.add(Dropout(dropout_rate))
          model.add(GRU(units = num_lstm_nodes, return_sequences = False))
        elif num_hidden_layers == 3:
          model.add(GRU(units = num_lstm_nodes, return_sequences = True, input_shape = (timesteps, 5)))
          model.add(Dropout(dropout_rate))
          model.add(GRU(units = num_lstm_nodes, return_sequences = True))
          model.add(Dropout(dropout_rate))
          model.add(GRU(units = num_lstm_nodes, return_sequences = False))
    
    model.add(Dropout(dropout_rate))
    model.add(Dense(units = 1))   

    if (approach == 240) | (approach ==31):
        model.add(Activation('sigmoid'))
        model.compile(optimizer = RMSprop(lr=learning_rate, clipvalue=0.5),
            loss = 'binary_crossentropy',
            metrics=['acc','binary_crossentropy',tf.keras.metrics.AUC(name='auc')])
        
    elif (approach == 63) | (approach == 241):
        model.add(Activation('linear'))
        model.compile(optimizer = Adam(lr=learning_rate, clipvalue=0.5),
            loss = 'mean_squared_error',
            metrics=[tf.keras.metrics.RootMeanSquaredError(name='RMSE'),
                     'mean_squared_error',
                     tf.keras.metrics.MeanAbsoluteError(name='MAE', dtype=None)])  
    return model  


# run time series validation, returing the average validation loss (out of sample validation)
def timeseriesCV(dataset, X,y,y_df,fold_size, numberofdays, timesteps, learning_rate,num_hidden_layers, num_lstm_nodes, num_batch, dropout_rate, number_of_folds, approach, forecastdays, arch):
  metric_list = []
  test_auc = []
  test_acc = []
  test_balacc = []
  test_RMSE = []
  test_MSE = []
  test_MAE = []

  test_balacc = []
  if approach == 63:
    foldlimit = int(number_of_folds/3)
  else:
    foldlimit = int((number_of_folds-3)*0.6)+1

  for foldindex in range(1,foldlimit):

    X_train, X_val, X_test, y_train, y_val, y_test, y_train_df, y_val_df, y_test_df  = createfolds(dataset, X, y, y_df, fold_size, foldindex, numberofdays, timesteps, approach, forecastdays)
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 5))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 5))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 5))
    
    if approach == 240:
      class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(y_train),
                                                  y_train)
      class_weights = dict(enumerate(class_weights))
    if arch == 'LSTM':
        model = create_lstm_model(timesteps=timesteps,
                            learning_rate = learning_rate,
                            num_hidden_layers = num_hidden_layers,
                            num_lstm_nodes= num_lstm_nodes,
                            dropout_rate = dropout_rate,
                            approach = approach)
    elif arch == 'GRU':
        model = create_GRU_model(timesteps=timesteps,
                    learning_rate = learning_rate,
                    num_hidden_layers = num_hidden_layers,
                    num_lstm_nodes= num_lstm_nodes,
                    dropout_rate = dropout_rate,
                    approach = approach)

    if (approach == 240) | (approach == 31):
        es = EarlyStopping(monitor='val_auc', 
                           mode='max', 
                           verbose=0, 
                           patience=10)
        
        checkpointer = ModelCheckpoint(filepath="weights.hdf5",
                                       monitor='val_auc',
                                       mode ='max',
                                       verbose=0, 
                                       save_best_only=True)
        
    elif (approach == 63) | (approach == 241):
        es = EarlyStopping(monitor='val_RMSE',
                           mode='min', 
                           verbose=1, 
                           patience=10)
        
        checkpointer = ModelCheckpoint(filepath="weights.hdf5",
                                       monitor = "val_RMSE",  #'val_mean_squared_error',
                                       mode ='min',
                                       verbose=1, 
                                       save_best_only=True)
    if approach == 240:
      history = model.fit(x=X_train,
                          y=y_train,
                          epochs=2,
                          batch_size=2**num_batch,
                          validation_data=(X_val,y_val),
                          verbose=0,
                          callbacks=[es, checkpointer],
                          class_weight = class_weights)

    else:
      history = model.fit(x=X_train,
                    y=y_train,
                    epochs=2,
                    batch_size=2**num_batch,
                    validation_data=(X_val,y_val),
                    verbose=0,
                    callbacks=[es, checkpointer])
    # calculate performance with test set
    y_test_pred = model.predict(X_test)

    if approach ==240:
      auc_hist = history.history['val_auc']
      auc_hist = [x for x in auc_hist if (not math.isnan(x))]
      auc = np.max(auc_hist)
      metric_list.append(auc)

      test_auc.append(roc_auc_score(y_test,y_test_pred))
      test_acc.append(accuracy_score(y_test,np.round(y_test_pred)))
      test_balacc.append(balanced_accuracy_score(y_test,np.round(y_test_pred)))

    elif (approach == 63) | (approach == 241):
      rmse_hist = history.history['val_RMSE']
      rmse_hist = [x for x in rmse_hist if (not math.isnan(x))]
      rmse = np.min(rmse_hist)
      metric_list.append(rmse)     

      test_RMSE.append(np.sqrt(mean_squared_error(y_test,y_test_pred)))
      test_MSE.append(mean_squared_error(y_test,y_test_pred))
      test_MAE.append(mean_absolute_error(y_test,y_test_pred))

    del model
    K.clear_session()
    
  metric_list = [x for x in metric_list if (not math.isnan(x))]
  av_metric = np.average(metric_list)
  
  if approach == 240:
    global best_test_AUC 

    if av_metric > best_test_AUC:
      best_test_AUC = av_metric
      print('AUC: ', np.average(test_auc))
      test_perf_df['auc'] = np.average(test_auc)
      best_test_accuray =  np.average(test_acc)
      print('Accuracy: ', best_test_accuray)
      test_perf_df['acc'] = best_test_accuray
      best_test_balaccuray = np.average(test_balacc)
      print('Balcanced Accuracy: ', best_test_balaccuray)
      test_perf_df['balacc'] = best_test_balaccuray
      test_perf_dir = '/content/drive/My Drive/rep_code/hyperparamopt_test_performance_' + arch + str(approach)+'_d_'+str(forecastdays) +'.csv'
      with open(test_perf_dir, 'w') as csv_file:
        test_perf_df.to_csv(path_or_buf=csv_file,  index=False)

  else:
    global best_test_RMSE
    if av_metric < best_test_RMSE:
      best_test_RMSE = av_metric
      print('RMSE: ', np.average(test_RMSE))
      test_perf_df['RMSE'] = np.average(test_RMSE)

      best_test_MSE =  np.average(test_MSE)
      print('MSE: ', best_test_MSE)
      test_perf_df['MSE'] = best_test_MSE

      best_test_MAE = np.average(test_MAE)
      print('MAE: ', best_test_MAE)
      test_perf_df['MAE'] = best_test_MAE

      test_perf_dir = '/content/drive/My Drive/rep_code/hyperparamopt_test_performance_'+ arch + str(approach)+'_d_'+str(forecastdays) +'.csv'
      with open(test_perf_dir, 'w') as csv_file:
        test_perf_df.to_csv(path_or_buf=csv_file,  index=False)
  return av_metric

# function to save the best parameter set for each delta_t
def saveoptresults(search_result, parameter_df, forecastdays, approach):

  learning_rate = search_result.x[0]
  num_hidden_layers = search_result.x[1]
  num_lstm_nodes = search_result.x[2]
  num_batch = search_result.x[3]
  dropout_rate = search_result.x[4]
  
  parameter_df.loc[forecastdays,'auc'] = sorted(zip(search_result.func_vals, search_result.x_iters))[0][0]
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
    firstfold = int((number_of_folds-3)*0.6)+1
    foldlimit = firstfold+2

  testfolds = range(firstfold,foldlimit)
  for foldindex in testfolds:

    X_train, X_val, X_test, y_train, y_val, y_test, y_train_df, y_val_df, y_test_df  = createfolds(dataset, X, y, y_df, fold_size, foldindex, numberofdays, timesteps, approach, forecastdays)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 5))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 5))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 5))

    if approach == 240:
      class_weights = class_weight.compute_class_weight('balanced',
                                                np.unique(y_train),
                                                y_train)
      class_weights = dict(enumerate(class_weights))
    if arch == 'LSTM':
        model = create_lstm_model(timesteps=timesteps,
                            learning_rate = learning_rate,
                            num_hidden_layers = num_hidden_layers,
                            num_lstm_nodes= num_lstm_nodes,
                            dropout_rate = dropout_rate,
                            approach = approach)
    elif arch == 'GRU':
        model = create_GRU_model(timesteps=timesteps,
                    learning_rate = learning_rate,
                    num_hidden_layers = num_hidden_layers,
                    num_lstm_nodes= num_lstm_nodes,
                    dropout_rate = dropout_rate,
                    approach = approach)

    if approach == 240:
        es = EarlyStopping(monitor='val_auc', 
                           mode='max', 
                           verbose=0, 
                           patience=10)
        
        checkpointer = ModelCheckpoint(filepath="weights.hdf5",
                                       monitor='val_auc',
                                       mode ='max',
                                       verbose=0, 
                                       save_best_only=True)
        
    elif (approach == 63) | (approach == 241):
        es = EarlyStopping(monitor='val_RMSE', 
                           mode='min', 
                           verbose=1, 
                           patience=10)
        
        checkpointer = ModelCheckpoint(filepath="weights.hdf5",
                                       monitor='val_RMSE',
                                       mode ='min',
                                       verbose=1, 
                                       save_best_only=True)



    if (approach == 240) | (approach == 31):
      history = model.fit(x=X_train,
                          y=y_train,
                          epochs=2,
                          batch_size=2**num_batch,
                          validation_data=(X_val,y_val),
                          verbose=1,
                          callbacks=[es, checkpointer],
                          class_weight = class_weights)

    else:
      history = model.fit(x=X_train,
                    y=y_train,
                    epochs=2,
                    batch_size=2**num_batch,
                    validation_data=(X_val,y_val),
                    verbose=0,
                    callbacks=[es, checkpointer])

    if approach == 240:
      print("AUC", np.max(history.history['val_auc']))
    else:
      print("RMSE", np.min(history.history['val_RMSE']))
    
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
    
    # Create the neural network with a set of hyper-parameters + run time series validation     
    av_metric = timeseriesCV(dataset, X,y,y_df,fold_size, numberofdays, timesteps, learning_rate,num_hidden_layers, num_lstm_nodes, num_batch, dropout_rate, number_of_folds, approach, forecastdays, arch)
    if (approach == 240) | (approach == 31):
      print()
      print("Average AUC: ", (av_metric))
      print()
    else: 
      print()
      print("Average RMSE: ", (av_metric))
      print()      
    if (approach == 240 | approach == 31):
      return -av_metric
    else:
      return av_metric
    

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
  
  # Bayesian optimization
  search_result = gp_minimize(func=fitness,
                              dimensions=dimensions,
                              acq_func='EI', # Expected Improvement.
                              n_calls=11,
                              x0=default_parameters)
  
  # store best parameters and optimization results
  parameter_df = saveoptresults(search_result, parameter_df, forecastdays, approach)
  
  # store predictions for test data
  test_df = savetestresults(dataset, X, y, y_df,approach, fold_size, number_of_folds, forecastdays, numberofdays, timesteps, search_result)
