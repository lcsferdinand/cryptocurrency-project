import math as m
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

def garch_df(df_sg,p,q,test_ratio=0.25): #make data frame base on GARCH(p,q)
  # df_sg = pd.DataFrame(u2)
  # df_sg.rename(columns={'Return':'u2'},inplace=True)
  # df_sg['vol_prox']=(df['Return'].iloc[:-1]-df['Return'].iloc[:-1].mean())**2

  if p>q:
    #return
    for i in range(1,p):
      df_sg['u2_shift'+str(i)] =  df_sg['u2'].shift(-i)

    #volatility
    diff = p-q
    for i in range(p,diff-1,-1):
      df_sg['vol_prox_shift'+str(i)]= df_sg['vol_prox'].shift(-i)

  elif p<q:
    #return
    diff = q-p
    for i in range(q-1,diff-1,-1):
      df_sg['u2_shift'+str(i)] =  df_sg['u2'].shift(-i)

    #volatility
    for i in range(q,0,-1):
      df_sg['vol_prox_shift'+str(i)]= df_sg['vol_prox'].shift(-i)
      
  else:
    #return
    for i in range(1,p):
      df_sg['u2_shift'+str(i)] =  df_sg['u2'].shift(-i)
    #volatility
    for i in range(1,q+1):
      df_sg['vol_prox_shift'+str(i)]= df_sg['vol_prox'].shift(-i)  

  df_sg = df_sg.iloc[:-max(p,q)]
  df_sg = df_sg[sorted(df_sg.columns)]

  if p<q:
    selected_col = df_sg[sorted(df_sg.columns)].iloc[:,:p+1].iloc[:,-p:].columns #select return
    selected_col= selected_col.append(df_sg[sorted(df_sg.columns)].iloc[:,-q-1:-1].columns) #select volatility
  else:
    selected_col = df_sg[sorted(df_sg.columns)].iloc[:,:p].iloc[:,-p:].columns #select return
    selected_col= selected_col.append(df_sg[sorted(df_sg.columns)].iloc[:,-q-1:-1].columns) #select volatility
    
  n_test = m.floor(len(df_sg)*test_ratio)
  # print('n_test: ',n_test)
  X_sg = df_sg[selected_col].iloc[:-n_test]
  y_sg = df_sg[df_sg.iloc[:,-1:].columns].iloc[:-n_test]
  X_sg_val = df_sg[selected_col].iloc[-n_test:]
  y_sg_val = df_sg[df_sg.iloc[:,-1:].columns].iloc[-n_test:]

  return X_sg,y_sg,X_sg_val,y_sg_val

def calculate_aic(n, mse, num_params):
	  aic = n * m.log(mse) + 2 * num_params
	  return aic

def calculate_bic(n, mse, num_params):
	  bic = n * m.log(mse) + num_params * m.log(n)
	  return bic