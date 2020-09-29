# -*- coding: utf-8 -*-


import yfinance as yf
import lxml
import pandas as pd
import numpy as np


def getsp500data(numberofstocks, startdate, enddate):
    r""" Get daily stock data (Date, Open, High, Low, Close, Volume, Dividends,
    Stock Splits, Stock Symbol) for S&P 500 stocks with the highest trade
    volume between start and end
    
    Parameters
    ----------
    numberofstocks : number of stocks with highest trade volume
    
    start : start date of data extraction period
    
    end: end data of data extraction period
    
    Returns
    -------
    stockdata : DataFrame of S&P 500 stocks with highest trade volume between 
    start and end with columns: Date, Open, High, Low, Close, Volume, 
    Dividends, Stock Splits, Stock 
    
    """
    #extract the list of SP 500 companies from wikipedia
    wikidata = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    sp500table = wikidata[0]
    sp500table.head
    sp500_stock_symbols = sp500table['Symbol']
    sp500_stock_symbols=sp500_stock_symbols.str.replace(".","-")
    stock_symbol = []
    tradevolume = []  
    
    for symbol_index in sp500_stock_symbols:
        tickerData = yf.Ticker(symbol_index)
        tickerDf = tickerData.history(period='1d', start=startdate, end=enddate)
        totalsum = np.sum(tickerDf['Volume'])
        stock_symbol.append(symbol_index)
        tradevolume.append(totalsum)     
        
    tradevolumedf = {'Stock':stock_symbol,'Volume':tradevolume}
    tradevolumedf = pd.DataFrame(tradevolumedf)
    volume_ordered = np.array(tradevolume)
    volume_ordered = np.sort(volume_ordered)
    tradevolumemaxdf = tradevolumedf[tradevolumedf['Volume']>=volume_ordered[-numberofstocks]]
    volumeofmaxstocks = tradevolumemaxdf['Stock']
    
    firststock = volumeofmaxstocks.iloc[0]
    tickerData = yf.Ticker(firststock)
    stockdata = tickerData.history(period='1d', start=startdate, end=enddate)
    stockdata['Stock'] = firststock
    volumeofstocks = volumeofmaxstocks[1:]
    
    for symbol_index in volumeofstocks:
        tickerData = yf.Ticker(symbol_index)
        tickerDf = tickerData.history(period='1d', start=startdate, end=enddate)
        tickerDf['Stock'] = symbol_index
        stockdata = pd.concat([stockdata,tickerDf],ignore_index=False)

    stockdata.reset_index(level=0, inplace=True)
    
    return stockdata
   

def stockfullhistcheck(stockdata):
    r""" Check whether the history is complete for each stock
    
    Parameters
    ----------
    stockdata : stock data
    
    
    Returns
    -------

    """
    
    for stockindex in np.unique(stockdata['Stock']):
        if np.min(stockdata[stockdata['Stock']==stockindex]['Date']) > np.min(stockdata['Date']):
            print("Stock listed after start date: ", stockindex)
            print(np.min(stockdata[stockdata['Stock']==stockindex]['Date']))
            print(" ")
            #print(np.max(stockdata[stockdata['Stock']==stockindex]['Date']))
        elif np.max(stockdata[stockdata['Stock']==stockindex]['Date']) < np.max(stockdata['Date']):
            print("Stock removed from S&P 500 before end date: ", stockindex)
            print(np.max(stockdata[stockdata['Stock']==stockindex]['Date']))
            print(" ")
        
    return



