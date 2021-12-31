import os
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