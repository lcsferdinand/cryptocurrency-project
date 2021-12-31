import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import plotly.graph_objects as go
import math as m
from scipy import stats
import seaborn as sns
from itertools import combinations
from arch import arch_model
import yfinance as yf
from statsmodels.stats import diagnostic
import statsmodels.api as sm

#Machine Learning
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR

def stat_desc(y_pred):
    print('Skewness {}'.format(stats.skew(y_pred)))
    print('Kurtosis {}'.format(stats.kurtosis(y_pred)))
    print('Mean {}'.format(y_pred.mean()))
    print('STD {}'.format(y_pred.std()))

def calculate_aic(n, mse, num_params):
	  aic = n * m.log(mse) + 2 * num_params
	  return aic

def calculate_bic(n, mse, num_params):
	  bic = n * m.log(mse) + num_params * m.log(n)
	  return bic

class data:
  def coin(self):
    self.btc = yf.download("BTC-USD",start = '2017-08-01',end='2021-09-30')
    self.eth = yf.download("ETH-USD",start = '2017-08-01',end='2021-09-30')
    self.tether = yf.download("USDT-USD",start = '2017-08-01',end='2021-09-30')
    self.bnb = yf.download("BNB-USD",start = '2017-08-01',end='2021-09-30')
    print('coins data sucessfully generated')
    
    for coins in self.btc,self.eth,self.tether,self.bnb:
      coins.reset_index(inplace=True)
      coins['Date'] = pd.to_datetime(coins['Date'])
      #Return
      index = list(range(0, len(coins)))
      coins['index_dummy'] = index
      coins.set_index("index_dummy", inplace = True)

      coins['Return'] = 0.0
      for i in range (len(coins['Close'])-1):
        coins['Return'][i]=m.log(coins['Close'][i+1]/coins['Close'][i])*100
      globals()['self.%s' % coins] = coins.iloc[:-1]

    return self

  def stat_desc(self):
    print('BTC')
    print('Skewness {}'.format(stats.skew(self.btc['Return'])))
    print('Kurtosis {}'.format(stats.kurtosis(self.btc['Return'])))
    print('Mean {}'.format(self.btc['Return'].mean()))
    print('STD {}'.format(self.btc['Return'].std()))
    print('\n')
    print('ETH')
    print('Skewness {}'.format(stats.skew(self.eth['Return'])))
    print('Kurtosis {}'.format(stats.kurtosis(self.eth['Return'])))
    print('Mean {}'.format(self.eth['Return'].mean()))
    print('STD {}'.format(self.eth['Return'].std()))
    print('\n')
    print('tether')
    print('Skewness {}'.format(stats.skew(self.tether['Return'])))
    print('Kurtosis {}'.format(stats.kurtosis(self.tether['Return'])))
    print('Mean {}'.format(self.tether['Return'].mean()))
    print('STD {}'.format(self.tether['Return'].std()))
    print('\n')
    print('BNB')
    print('Skewness {}'.format(stats.skew(self.bnb['Return'])))
    print('Kurtosis {}'.format(stats.kurtosis(self.bnb['Return'])))
    print('Mean {}'.format(self.bnb['Return'].mean()))
    print('STD {}'.format(self.bnb['Return'].std()))
    return self

  def split(self):
    for coins in self.btc,self.eth,self.tether,self.bnb:
      self.btc_n = self.btc.iloc[:945]
      self.btc_p = self.btc.iloc[945:]
      self.eth_n = self.eth.iloc[:945]
      self.eth_p = self.eth.iloc[945:]
      self.tether_n = self.tether.iloc[:945]
      self.tether_p = self.tether.iloc[945:]
      self.bnb_n = self.bnb.iloc[:945]
      self.bnb_p = self.bnb.iloc[945:]
    return self

  def return_value(self):
    self.ret_btc_n = self.btc_n['Return']
    self.ret_btc_p = self.btc_p['Return']
    self.ret_eth_n = self.eth_n['Return']
    self.ret_eth_p = self.eth_p['Return']
    self.ret_tether_n = self.tether_n['Return']
    self.ret_tether_p = self.tether_p['Return']
    self.ret_bnb_n = self.bnb_n['Return']
    self.ret_bnb_p = self.bnb_p['Return']
    return self

class model:
  def model_data(self, ret_data):
    n_test = m.floor(len(ret_data)/4)
    self.X = ret_data.iloc[:-n_test]
    self.y = ret_data.shift(-1).iloc[:-n_test]
    self.X_val = ret_data.iloc[-n_test:-1]
    self.y_val = ret_data.shift(-1).iloc[-n_test:-1]

  def svr(self, kernel, eps, C=3, gamma=3, degree=3):
    X = self.X#.reshape(-1,1)
    y = self.y#.reshape(-1,1)
    X_val = self.X_val#.reshape(-1,1)
    y_val = self.y_val#.reshape(-1,1)
    regressor = SVR(kernel = kernel, C=C, epsilon=eps,gamma=gamma,degree=degree)
    regressor.fit(X,y)

    #Predicting a new result
    score = regressor.score(X,y)
    y_pred = regressor.predict(X_val)
    score_train = regressor.score(X,y)
    score_test = regressor.score(X_val,y_val)
    self.score = score
    self.y_pred = y_pred
    self.score_train = score_train
    self.score_test = score_test
    
    #SVR Function
    alpha = regressor.dual_coef_
    support_vector = regressor.support_vectors_
    if kernel =='poly':
      print('w coef: {}'.format(np.matmul(alpha,(gamma**degree)*(support_vector)**degree)))
      print('b coef: {} \n'.format(regressor.intercept_))
      self.w_coef = np.matmul(alpha,(gamma**degree)*(support_vector)**degree)
      self.b_coef = regressor.intercept_

    else:
      print('w coef: {}'.format(np.matmul(alpha,support_vector)))
      print('b coef: {} \n'.format(regressor.intercept_))
      self.w_coef = np.matmul(alpha,support_vector)
      self.b_coef = regressor.intercept_

    #Model Validation
    print("Train score: {}".format(score_train))
    print('Test score: {} \n'.format(score_test))
    print("MSE:", mean_squared_error(y_val,y_pred))
    print("MAE:",mean_absolute_error(y_val,y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_val,y_pred)))
    print('AIC {}'.format(calculate_aic(len(X_val),mean_squared_error
                                        (y_val,y_pred,),2)))
    print('BIC {} \n'.format(calculate_bic(len(X_val),mean_squared_error
                                        (y_val,y_pred,),2)))
    print('Statistika Deskriptif Data Populasi')
    print(stat_desc(y_val))
    print('Statistika Deskriptif Prediksi')
    print(stat_desc(y_pred))

    #Plot
    figure(figsize=(10, 6), dpi=80)
    plt.plot(y_val,label='Observed volatility');
    plt.plot(pd.Series(y_pred, index=y_val.index),'r',ls='--',label='SVR forecasted volatility');
    plt.title('Predicted Volatility');
    plt.xlabel('Days')
    plt.ylabel('Volatility')
    plt.legend(loc='upper right');

    return y_pred, regressor