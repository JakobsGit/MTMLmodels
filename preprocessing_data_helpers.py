# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def replacenans(dataset):
    r"""
    Replaces all NaN in stock data DataFrame with data from day before
    
    Parameters
    ----------
    dataset : DataFrame output from getsp500data()
        Containing the following Columns:
            Open, High, Low, Volume

    Returns
    -------
    None.

    """
    
    dataset[dataset['Open'].isna()==True]
    naindex = dataset.index[dataset['Open'].isna()==True]
    
    
    for i in naindex:
        dataset.loc[i,'Open'] = float(dataset.loc[(i-1),'Open'])
        dataset.loc[i,'High']=float(dataset.loc[i-1,'High'])
        dataset.loc[i,'Low']=float(dataset.loc[i-1,'Low'])
        dataset.loc[i,'Close']=float(dataset.loc[i-1,'Close'])
        dataset.loc[i,'Volume']=float(dataset.loc[i-1,'Volume'])

    return



def createreturncolumn(dataset,n, approach):
    r"""
    Creates a new column containing returns for a DataFrame with stock data
    
    Parameters
    ----------
    dataset : DataFrame containing the Columns Close and Stock
        
    n: Parameter to calculate stock return after n trade days
    approach: triggers different return calculations 

    Returns
    -------
    dataset : DataFrame
        Input DataFrame with additional Return Columns

    """

    dataset['Return']= np.zeros(dataset.shape[0]) 
    dataset['Return'] = dataset.shift(-n)['Close'].values
    
    dataset['1dayReturn']= np.zeros(dataset.shape[0])
    dataset['1dayReturn'] = dataset.shift(-1)['Close'].values
    
    rowstobedeleted = []  
    for stockindex in np.unique(dataset['Stock']):
            for i in range(-n,0):
                rowstobedeleted.append(dataset.index[dataset['Stock']==stockindex][i])   
    rowstobedeleted = np.array(rowstobedeleted)    
    dataset = dataset.drop(rowstobedeleted, axis=0)
    
    dataset.reset_index(level=0, inplace=True)
    dataset = dataset.drop(columns='index')
    
    
    if (approach == 240) | (approach == 31):
        dataset['Return'] = dataset['Return']/dataset['Close']
        dataset['Return'] = dataset['Return']-1
        
        dataset['1dayReturn'] = dataset['1dayReturn']/dataset['Close']
        dataset['1dayReturn'] = dataset['1dayReturn']-1
    
    elif approach == 63:
        dataset['Return'] = dataset['Return']-dataset['Close']
        
        dataset['1dayReturn'] = dataset['1dayReturn']-dataset['Close']
    
    return dataset


def createtargetcolumn(dataset, approach): 
    r"""
    approach = 240 or approach = 31:
    Creates Target column indicating if the return of a stock is higher
    than the median return of the set of stocks for each day
    
    approach = 63:  
    creates target column containing the absolute return 
    (price at time point t - price at time point t-n)
    
    Parameters
    ----------
    dataset : DataFrame with stock data containing the following columns:
        Date, Return
    
    approach: triggers different target calculations

    Returns
    -------
    dataset : DataFrame with additional column Target
    
    """
    if (approach == 240) | (approach == 31):
        targetvec = np.zeros(dataset.shape[0])
        for dateindex in np.unique(dataset['Date']):
            returnsofday = dataset[dataset['Date']==dateindex]['Return']
            returnsofday = np.array(returnsofday)
        
            returns_ordered = np.sort(returnsofday)
            mediandayreturn = np.median(returns_ordered)
            indicesofday = dataset.index[dataset['Date']==dateindex]
            greatermedian= dataset.loc[indicesofday]['Return']>= mediandayreturn
            greatermedian[greatermedian]=1
            greatermedian = np.array(greatermedian)
            targetvec[indicesofday]= greatermedian
            dataset['Target'] = targetvec
        dataset.reset_index(level=0, inplace=True)
        dataset = dataset.drop(columns='index')
    elif approach == 63:
        dataset['Target'] = dataset['Return']        
    
    return dataset




def deletedividendentries(dataset):
    r"""
    deletes all entries for a date of a stock data DataFrame that contain 
    only dividend information and no price movements 

    Parameters
    ----------
    dataset : DataFrame with stock data containing the following columns:
        Date, Stock, Close, Dividends

    Returns
    -------
    dataset : Cleaned DataFrame

    """
    
    limitnumber = np.unique(dataset['Stock']).shape[0]*0.2
    for dateindex in np.unique(dataset['Date']):
        if dataset[dataset['Date']==dateindex].shape[0] < limitnumber:
            print(dateindex)
            #print(dateindex)
            #print(dataset[dataset['Date']==dateindex].shape[0])
            rowindex = dataset.index[dataset['Date']==dateindex]
            for i in rowindex:
              print(i)
              if float(dataset.loc[i]['Close']) == float(dataset.loc[i-1]['Close']):
                  #print(dataset.loc[rowindex])
                  #print(dataset.loc[rowindex]['Dividends'])
                  #print(dataset.loc[rowindex-1])
                  dataset = dataset.drop(i, axis=0)
                  dataset.reset_index(level=0, inplace=True)
                  dataset = dataset.drop(columns='index')
    return dataset
    

def calculateXdayReturn(onedayreturnfactor,timesteps, day,n):
    r"""
    calculates the accumulated return of the last day trading days

    Parameters
    ----------
    onedayreturnfactor : contains the returns of a stock +1 
    timesteps: number of total trading days
    day: number of accumulated trading days

    Returns
    -------
    ret : accumulated return for day trading days

    """
    start = timesteps-day
    ret = np.product(onedayreturnfactor[start:timesteps]) 
    ret = ret -1
    return ret


def createXseries(X, one_stock_data, approach, timesteps, n):
    r"""
    creates X containing time series of return data 

    Parameters
    ----------
    X: variable for storing time series
    one_stock_data : contains the returns of a stock
    approach: different elements are saved in time series
    timesteps: number of total trading days
    n: return predicted after n days

    Returns
    -------
    

    """
    
    if (approach == 240):
        featureindices = np.array(range(-timesteps-n,-n,1))
        returns = np.array(one_stock_data['1dayReturn'])
        X.append(returns[featureindices])        

    elif (approach == 63):
        featureindices = np.array(range(-timesteps-n,-n,1))
        returns = np.array(one_stock_data['Return'])
        X.append(returns[featureindices])
    
    elif (approach == 31):
        features31 = []
        onedayreturnfactor = np.array(one_stock_data['1dayReturn'])+1
        featurevector = [240,220,200,180,160,140,120,100,80,60,40,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
        for day in featurevector:
            features31.append(calculateXdayReturn(onedayreturnfactor,timesteps,day,n))
        X.append(features31)
        
        
def createseries(dataset, timesteps, n, approach):
    r"""
    uses stock data to create X containing return timeseries, y containing targets to be forecasted, 
    y_df containing raw data about the targets (returns, date, stockname)

    Parameters
    ----------
    dataset: dataset containing stock data
    timesteps : trading days taken into account to create time series
    approach: different elements are saved in time series
    n: return predicted after n days

    Returns
    -------
    X: return time series
    y: targets to be forecasted 
    y_df: data about the targets (returns, date, stockname)

    """
    
    dataset = dataset.drop(columns = ['Open','High','Low', 'Volume', 'Dividends', 'Stock Splits'])

            
    dataset = dataset.reset_index(level=0, drop=True)


    X = []
    y = []
    y_df = dataset.iloc[0:1,:]
    
    endindex = np.unique(dataset['Date']).shape[0]
    for i in range(timesteps+n, endindex):
        #print(i, ' / ', endindex)
        perioddata = dataset[dataset.Date >=np.unique(dataset.Date)[i-(timesteps+n)]]
        datelimit = np.unique(dataset.Date)[i]
        
        perioddata = perioddata[perioddata.Date <= datelimit]                                  
        for stockindex in np.unique(perioddata['Stock']):
            one_stock_data = perioddata[perioddata.Stock==stockindex]
            if one_stock_data.shape[0]>=timesteps+n+1:
                
                createXseries(X, one_stock_data, approach, timesteps, n)
                
                y_vec = np.array(one_stock_data['Target'])
                y.append(float(y_vec[-1]))
                
                y_df1 = one_stock_data.iloc[timesteps+n:timesteps+n+1,:]
                y_df = pd.concat([y_df,y_df1],ignore_index=True)
                #print(stockindex)
                #print(one_stock_data.shape)
    X = np.array(X)
    y = np.array(y)
    y_df = y_df.drop(0, axis=0)
        
    y_df = y_df.reset_index(level=0, drop=True)
    
    return X, y, y_df

 

def standardize_input(y_train_df, dataset, X_train, X_val, X_test, timesteps):
    r"""
    standardizes X_test, X_val and X_test
    
    Parameters
    ----------
    y_train_df: data about the training targets (returns, date, stockname)
    timesteps : trading days taken into account to create time series
    dataset: dataset containing stock data
    
    X_train: time series data for training
    X_val:  time series data for validation
    X_test: time series data for testing
    
    Returns
    -------
    X_test: standardized X_test
    X_val:  standardized X_val
    X_test: standardized X_test
    
    """
    
    startdate = np.min(y_train_df.Date)
    alldates = np.unique(dataset.Date).astype('datetime64[s]')
    outputindexstart = int(np.where(alldates == startdate)[0])
    
    onlyinputdata = dataset[(dataset['Date']< startdate) & (dataset['Date'] >= alldates[outputindexstart-timesteps])] 
    
    
    onlyinputdata = onlyinputdata.drop(columns=['Open', 	'High', 	'Low', 'Volume', 'Dividends', 'Stock Splits'])
    
    trainingdatapoints = pd.concat([onlyinputdata, y_train_df])
    
    trainingdatareturns = trainingdatapoints['1dayReturn']
    
    averagereturn = np.average(trainingdatareturns)
    stddevreturn = np.std(trainingdatareturns)  
    
    X_train[:,:] = X_train[:,:] - averagereturn
    X_train[:,:] = X_train[:,:]/stddevreturn
    
    X_val[:,:] = X_val[:,:] - averagereturn
    X_val[:,:] = X_val[:,:]/stddevreturn
    
    X_test[:,:] = X_test[:,:] - averagereturn
    X_test[:,:] = X_test[:,:]/stddevreturn    
    
    return X_train, X_val,  X_test
 





def createfolds(dataset, X, y, y_df, fold_size, foldindex, numberofdays, timesteps, approach):
    r"""
    creates folds for time series validation
    
    Parameters
    ----------
    y_train_df: data about the training targets (returns, date, stockname)
    
    dataset: dataset containing stock data
    
    X: time series data 
    y: target data
    y_df: additional data about targets (date, return, stockname)
    fold_size: number of trading days in one fold
    foldindex: number of fold
    timesteps : trading days taken into account to create time series
    approach: number of folds differs according to approach
 
    Returns
    -------
    X_train, 
    X_val, 
    X_test: train, validation and test folds
    y_train, 
    y_val, 
    y_test: train, validation and test targets 
    y_train_df, 
    y_val_df, 
    y_test_df: additional information about train, validation and test targets    
    
    """

    startindex = fold_size*(foldindex-1)
    startdate = np.unique(y_df.Date)[startindex] 
    if approach == 63:
      train_chunk_date_index = int(fold_size*(7))+ startindex
    else:
      train_chunk_date_index = int(fold_size*(3))+ startindex
    train_chunk_max_date = np.unique(y_df.Date)[train_chunk_date_index]   
            
    train_indices = y_df.index[(y_df.Date<train_chunk_max_date) & (y_df.Date>=startdate)]
    X_train = X[train_indices,:]
    y_train = y[train_indices]
    y_train_df = y_df.loc[train_indices,:]

           
    val_date_end_index = train_chunk_date_index + fold_size
    val_chunk_max_date = np.unique(y_df.Date)[val_date_end_index]         
    val_indices = y_df.index[(y_df.Date<val_chunk_max_date) & (y_df.Date>=train_chunk_max_date)]

            
    X_val = X[val_indices,:]
    y_val = y[val_indices]
    y_val_df = y_df.loc[val_indices,:]
    

    test_date_end_index = val_date_end_index+fold_size-1  #+ numberofdays%(number_trade_periods + number_of_folds)-1
    test_chunk_max_date = np.unique(y_df.Date)[test_date_end_index]         
    test_indices = y_df.index[(y_df.Date<test_chunk_max_date) & (y_df.Date>=val_chunk_max_date)]

    
    X_test = X[test_indices,:]
    y_test = y[test_indices]
    y_test_df = y_df.loc[test_indices,:]
    
    if approach == 240:
        X_train, X_val, X_test = standardize_input(y_train_df, dataset, X_train, X_val, X_test, timesteps)


            
    return X_train, X_val, X_test, y_train, y_val, y_test, y_train_df, y_val_df, y_test_df      




          