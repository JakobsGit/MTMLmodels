## Master's Thesis Code Repository


**get_data_functions.py**

* contains function that  gathers S&P 500 stock data with highest trade volume from yahoo finance

**preprocessing_data_helpers.py**

* contains functions that preprocess the data (remove "N/A", standardize data, feature and target generation,...)
	
**main_RF.py**
	
* contains functions that divide the prepared data set into training, validation and test sets and that run "cross validation for time series"
* Bayesian optimization to find the best hyper parameters (number of trees, max depth) for a random forest model (based on average validation loss)
* stores the results of the Bayesian optimization
* stores the best parameters for predicting the target in 20,10,5,3,2,1 trading days 
* uses the best parameters to predict targets of the test sets and stores the results in a data frame containing the date, return, and price information

Model: 
features: open, close, high, low prices, return
target: positive return -> 1, negative or zero return -> 0
model similiar to 
* [*C. Krauss et al. European Journal of Operational Research 259 (2017) 689–702. Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500](https://doi.org/10.1016/j.ejor.2016.10.031)
	
**main_LSTM.py**

* contains 2 different models: than can be chosen by the approach variable: 240(model similiar to **), 63 (model similiar to ***)
* contains functions that divide the prepared data set into training, validation and test sets and that run "cross validation for time series"
* Bayesian optimization to find the best hyper parameters (learning rate, hidden layers, lstm nodes, batch size, dropout rate) for a LSTM model (based on average validation loss)	
* stores the results of the Bayesian optimization
* stores the best parameters for predicting the target in 20,10,5,3,2,1 trading days 
* uses the best parameters to predict targets of the test sets and stores the results in a data frame containing the date, return, and price information
	
Model: 
features: open, close, high, low prices, return
target: positive return -> 1, negative or zero return -> 0
model according to 

* [**T. Fischer et al. European Journal of Operational Research 270 (2018) 654-669. Deep learning with long short-term memory networks for financial market predictions](https://doi.org/10.1016/j.ejor.2017.11.054)
* [***Univariate standard BiLSTM model (Uni) from N. N. Y. Vo et al. Decision Support Systems 124 (2019) 113097 Deep learning for decision making and the optimization of socially responsible investments and portfolio](https://doi.org/10.1016/j.dss.2019.113097)

+ GRU model (can be chosen by arch variable)

**main_GLM.py**

* contains 2 different models: than can be chosen by the reg variable: "lasso" or "ridge"
* contains functions that divide the prepared data set into training, validation and test sets and that run "cross validation for time series"
* Bayesian optimization to find the best hyper parameters (lambda) for a ridge/lasso regression	
* stores the results of the Bayesian optimization
* stores the best parameters for predicting the target in 20,10,5,3,2,1 trading days 
* uses the best parameters to predict targets of the test sets and stores the results in a data frame containing the date, return, and price information
	
Model: 
features: open, close, high, low prices, return
target: positive return -> 1, negative or zero return -> 0