# -*- coding: utf-8 -*-

import skopt
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from skopt import dump, load
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
import preprocessing_data_helpers
import get_data_functions

from get_data_functions import *
from preprocessing_data_helpers import *

from h2o.estimators.random_forest import H2ORandomForestEstimator
import math
import h2o
h2o.init()

randomseed = 1

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


approach = 31
timesteps = 20

# define parameter space
dim_num_trees = Integer(low=100, high=2000, name='ntrees')
dim_max_depth = Integer(low=5, high=20, name='max_depth')

dimensions_RF = [dim_num_trees,
              dim_max_depth]

# define default parameters to start the hyperparameter optimization
default_num_trees = 100
default_max_depth = 5

default_parameters_RF = [default_num_trees, default_max_depth]

number_of_folds = 15

best_test_AUC = 0.0
best_test_RMSE = 99999

# define list of delta_t, where delta_t return after delta_t trading days
nlist = [20,10,5,3,2,1]

#define parameter & loss results dataframe  
zeroph = np.zeros(len(nlist))
parameter_dict = {'auc':zeroph, 'ntrees':zeroph, 'max_depth':zeroph}
parameter_df_RF = pd.DataFrame(parameter_dict, columns = ['auc','ntrees','max_depth'], index = nlist) 

test_perf_dict = {'auc':np.zeros(1), 'acc':np.zeros(1),'balacc':np.zeros(1), 'RMSE':np.zeros(1), 'MSE':np.zeros(1),'MAE':np.zeros(1)}
test_perf_df = pd.DataFrame(test_perf_dict, columns = ['auc','acc','balacc','RMSE','MSE','MAE']) 


# get s&p 500 data with highest trade volume from yahoo finance 
stockdata = getsp500data(numberofstocks=50,startdate='2004-12-31', enddate='2019-12-31')  
stockfullhistcheck(stockdata)
replacenans(stockdata)

# convert input data to h2o frames
def createh2oframes(X, y):

    indexvec = np.arange(0,len(y))
    colnames = np.arange(-X.shape[1],0,1)
    
    df = pd.DataFrame(data=X[0:,0:],
            index=indexvec,    
            columns=colnames) 
    
    df['y'] = y
    hf = h2o.H2OFrame(df)
    if (approach == 31) | (approach == 240):
      hf['y'] = hf['y'].asfactor()
    
    return hf

# run time series validation, returing the average validation loss (out of sample validation)
def timeseriesCV_RF(dataset,X,y,y_df,fold_size, numberofdays, timesteps,ntrees,max_depth, number_of_folds, approach):
  metric_list = []
  test_auc = []
  test_acc = []
  test_balacc = []
  test_RMSE = []
  test_MSE = []
  test_MAE = []
  foldlimit = int((number_of_folds-3)*0.6)+1
  for foldindex in range(1,foldlimit):

    X_train, X_val, X_test, y_train, y_val, y_test, y_train_df, y_val_df, y_test_df  = createfolds(dataset, X, y, y_df, fold_size, foldindex, numberofdays, timesteps, approach, forecastdays)
    if (approach == 240) | (approach == 241):
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))

        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1]*X_val.shape[2]))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
        
    hf_train = createh2oframes(X_train, y_train)
    hf_val = createh2oframes(X_val, y_val)
    hf_test = createh2oframes(X_test, y_test)

    ydata= "y"
    xdata= hf_train.columns[:-1]

    rf_fit = H2ORandomForestEstimator(model_id='rf_fit', ntrees=int(ntrees),max_depth=int(max_depth), seed=randomseed, balance_classes=True)
    rf_fit.train(x=xdata, y=ydata, training_frame=hf_train, validation_frame = hf_val)

    y_test_pred = rf_fit.predict(hf_test)
    preddf = y_test_pred.as_data_frame(use_pandas=True)
    
    if (approach == 31) | (approach == 240):  
      y_test_pred = np.array(preddf['p1'])
    else:
      y_test_pred = np.array(preddf['predict'])
 

    if (approach == 31) | (approach == 240):
      auc = rf_fit.auc(valid=True) 
      metric_list.append(auc)

      test_auc.append(roc_auc_score(y_test,y_test_pred))
      test_acc.append(accuracy_score(y_test,np.round(y_test_pred)))
      test_balacc.append(balanced_accuracy_score(y_test,np.round(y_test_pred)))

    elif (approach == 63) | (approach == 241):
      rmse = rf_fit.rmse(valid = True)
      metric_list.append(rmse)

      test_RMSE.append(np.sqrt(mean_squared_error(y_test,y_test_pred)))
      test_MSE.append(mean_squared_error(y_test,y_test_pred))
      test_MAE.append(mean_absolute_error(y_test,y_test_pred))
    
  metric_list = [x for x in metric_list if (not math.isnan(x))]
  av_metric = np.average(metric_list)
  


  if (approach == 31) | (approach == 240):
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
      test_perf_dir = '/content/drive/My Drive/rep_code/hyperparamopt_test_performance_' + str(approach)+'_d_'+str(forecastdays) +'.csv'
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

      test_perf_dir = '/content/drive/My Drive/rep_code/hyperparamopt_test_performance_RF_' + str(approach)+'_d_'+str(forecastdays) +'.csv'
      with open(test_perf_dir, 'w') as csv_file:
        test_perf_df.to_csv(path_or_buf=csv_file,  index=False)

  if (approach == 240) | (approach == 31):
    av_metric = -av_metric


  return av_metric

# optimization function for the hyperparameter optimization
@use_named_args(dimensions=dimensions_RF)
def fitness_RF(ntrees,
            max_depth):

    # Print the hyper-parameters.
    print('ntrees:', ntrees)
    print('max_depth:', max_depth)
    print()

    av_auc = timeseriesCV_RF(dataset, X,y,y_df,fold_size, numberofdays, timesteps,ntrees,max_depth, number_of_folds, approach)
    print()
    print("Average Loss: ", -av_auc)
    print()

    return av_auc

# function to save the best parameter set for each delta_t
def saveoptresults(search_result, parameter_df, forecastdays, approach):
    
    ntrees = search_result.x[0]
    max_depth = search_result.x[1]    
      
    parameter_df.loc[forecastdays,'auc'] = sorted(zip(search_result.func_vals, search_result.x_iters))[0][0]
      
    parameter_df.loc[forecastdays,'ntrees'] = ntrees
    parameter_df.loc[forecastdays,'max_depth'] = max_depth
      
    parameterdfdir = 'parameter_df_RF_appr' + str(approach) +".csv"
    with open(parameterdfdir, 'w') as csv_file:
        parameter_df.to_csv(path_or_buf=csv_file,  index=False)
      
    optimizationdirect = 'optresult_RF_appr' + str(approach) + 'n' +str(forecastdays) +'.pkl'
    dump(search_result, optimizationdirect)
    return parameter_df

# train model with best hyperparameters and return the predictions for the test data (out of sample)
def savetestresults(dataset, X, y, y_df,approach, fold_size, number_of_folds, forecastdays, numberofdays, timesteps, ntrees, max_depth):
    
    firstfold = int((number_of_folds-3)*0.6)+1
    foldlimit = firstfold+3

    testfolds = range(firstfold,foldlimit)#range(int(number_of_folds*0.8-2),number_of_folds-2)

    for foldindex in testfolds:
      X_train, X_val, X_test, y_train, y_val, y_test, y_train_df, y_val_df, y_test_df  = createfolds(dataset, X, y, y_df, fold_size, foldindex, numberofdays, timesteps, approach, forecastdays)
      if (approach == 240) | (approach == 241):
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))

        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1]*X_val.shape[2]))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
        
      hf_train = createh2oframes(X_train, y_train)
      hf_val = createh2oframes(X_val, y_val)
      hf_test = createh2oframes(X_test, y_test)
    
      ydata= "y"
      xdata= hf_train.columns[:-1]
    
      rf_fit = H2ORandomForestEstimator(model_id='rf_fit', ntrees=int(ntrees),max_depth=int(max_depth), seed=randomseed, balance_classes=True)
      rf_fit.train(x=xdata, y=ydata, training_frame=hf_train, validation_frame=hf_val)
      
      pred = rf_fit.predict(hf_test)
      preddf = pred.as_data_frame(use_pandas=True)
      if (approach == 31) | (approach == 240):
        y_pred = np.array(preddf['p1'])

      else:
        y_pred = np.array(preddf['predict'])
    
      y_test_df['forecastdays'] = forecastdays
      y_test_df['fold'] = foldindex+4
    
      y_test_df['Prediction'] = y_pred
    
      if foldindex == testfolds[0]:
        test_df = y_test_df.copy()
      else:
        test_df = pd.concat([test_df, y_test_df])
     
    directory = 'test_df_RF_n_' +str(forecastdays) + 'appr'+ str(approach) +'.csv'
    with open(directory, 'w') as csv_file:
        test_df.to_csv(path_or_buf=csv_file,  index=False)
    
    return test_df


##########################################################
# HYPERPARAMETER OPTIMIZATION
##########################################################

for forecastdays in nlist:
  print("performance forecasted for ", forecastdays, "days")
  print(" ")
  
  #data preprocessing
  dataset = createreturncolumn(stockdata,forecastdays,approach)
  dataset = createtargetcolumn(dataset,approach)
  dataset = deletedividendentries(dataset)

  X, y, y_df = createseries(dataset, timesteps=timesteps, approach=approach, n=forecastdays)
  
  numberofdays = np.unique(y_df.Date).shape[0]
  fold_size = int(numberofdays/number_of_folds)
  
  # Bayesian optimization
  search_result = gp_minimize(func=fitness_RF,
                              dimensions=dimensions_RF,
                              acq_func='EI', # Expected Improvement.
                              n_calls=11,
                              x0=default_parameters_RF)
  
  # store best parameters and optimization results
  parameter_df_RF = saveoptresults(search_result, parameter_df_RF, forecastdays, approach)
  
  # store predictions for test data
  test_df = savetestresults(dataset, X, y, y_df,approach, fold_size, number_of_folds, forecastdays, numberofdays, timesteps, ntrees=100, max_depth=5)

