import math as m
from scipy import stats 
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def print_msg_box(msg, indent=1, width=None, title=None):
    """Print message-box with optional title."""
    lines = msg.split('\n')
    space = " " * indent
    if not width:
        width = max(map(len, lines))
    box = f'╔{"═" * (width + indent * 2)}╗\n'  # upper_border
    if title:
        box += f'║{space}{title:<{width}}{space}║\n'  # title
        box += f'║{space}{"-" * len(title):<{width}}{space}║\n'  # underscore
    box += ''.join([f'║{space}{line:<{width}}{space}║\n' for line in lines])
    box += f'╚{"═" * (width + indent * 2)}╝'  # lower_border
    print(box)

def stat_desc(y_pred):
    print('Skewness {}'.format(stats.skew(y_pred)))
    print('Kurtosis {}'.format(stats.kurtosis(y_pred)))
    print('Mean {}'.format(y_pred.mean()))
    print('STD {}'.format(y_pred.std()))

def garch_df(df,p,q,o,ratio=0.25): #make data frame base on GARCH(p,q)

  if p>q:
    #return
    for i in range(1,p):
      df['u_squared_shift'+str(i)] =  df['u_squared'].shift(-i)

    #volatility
    diff = p-q
    for i in range(p,diff-1,-1):
      df['vol_prox_shift'+str(i)]= df['vol_prox'].shift(-i)

  elif p<q:
    #return
    diff = q-p
    for i in range(q-1,diff-1,-1):
      df['u_squared_shift'+str(i)] =  df['u_squared'].shift(-i)

    #volatility
    for i in range(q,0,-1):
      df['vol_prox_shift'+str(i)]= df['vol_prox'].shift(-i)
      
  else:
    #return
    for i in range(1,p):
      df['u_squared_shift'+str(i)] =  df['u_squared'].shift(-i)
    #volatility
    for i in range(1,q+1):
      df['vol_prox_shift'+str(i)]= df['vol_prox'].shift(-i)  

  df = df.iloc[:-max(p,q)]
  df = df[sorted(df.columns)]

  if p<q:
    selected_col = df[sorted(df.drop('Return',axis=1).columns)].iloc[:,:p+1].iloc[:,-p:].columns #select return
    selected_col= selected_col.append(df[sorted(df.drop('Return',axis=1).columns)].iloc[:,-q-1:-1].columns) #select volatility
  else:
    selected_col = df[sorted(df.drop('Return',axis=1).columns)].iloc[:,:p].iloc[:,-p:].columns #select return
    selected_col= selected_col.append(df[sorted(df.drop('Return',axis=1).columns)].iloc[:,-q-1:-1].columns) #select volatility

  if o>0:
    df['I']=0
    for i in range(len(df['Return'])):
      if df['Return'][i] < 0:
        df['I'][i] = 1
      else:
        df['I'][i]=0

    for i in range(o):
      df['I_multiplied_shift'+str(i)] = df['I']*df[selected_col[i]]
      selected_col=np.append(selected_col,'I_multiplied_shift'+str(i))

    df.drop('I',axis=1,inplace=True)
    df.drop('Return',axis=1,inplace=True)

    n_test = m.floor(len(df)*ratio)
    X = df[selected_col].iloc[:-n_test]
    y = df[df.iloc[:,-(1+o):-o].columns].iloc[:-n_test]
    X_val = df[selected_col].iloc[-n_test:]
    y_val = df[df.iloc[:,-(1+o):-o].columns].iloc[-n_test:]

  else:
    n_test = m.floor(len(df)*ratio)
    X = df[selected_col].iloc[:-n_test]
    y = df[df.iloc[:,-1:].columns].iloc[:-n_test]
    X_val = df[selected_col].iloc[-n_test:]
    y_val = df[df.iloc[:,-1:].columns].iloc[-n_test:]
    
    df.drop('Return',axis=1,inplace=True)
    

  return X,y,X_val,y_val

def calculate_aic(n, mse, num_params):
	  aic = n * m.log(mse) + 2 * num_params
	  return aic

def calculate_bic(n, mse, num_params):
	  bic = n * m.log(mse) + num_params * m.log(n)
	  return bic