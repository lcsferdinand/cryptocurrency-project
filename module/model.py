class model:
  def train_test_split(self, ret_data,test_ratio=0.25):
    """
    ret_data: return of cryptocurrency data
    """
    n_test = m.floor(len(ret_data)*test_ratio)
    self.ret_data = ret_data
    self.X = ret_data.iloc[:-n_test]
    self.y = ret_data.shift(-1).iloc[:-n_test]
    self.X_val = ret_data.iloc[-n_test:-1]
    self.y_val = ret_data.shift(-1).iloc[-n_test:-1]

  def svr(self, coins, period, kernel, eps, C=3, gamma=3, degree=3,plot_svr=True,desc_stat_svr=True,rv='r'):
    """
    kernel: svr kerenel
    eps: svr epsilon 
    C: svr C
    gamma: svr gamma
    degree: svr degree
    plot_svr: if True then the plot will be shown
    rv: calculate return or volatiltiy, where r: return and v: volatiliy
    first run rv = 'r' to able to run rv = 'v'
    """
    if rv == 'r':
      ret_data = self.ret_data
      X = self.X#.reshape(-1,1)
      y = self.y#.reshape(-1,1)
      X_val = self.X_val#.reshape(-1,1)
      y_val = self.y_val#.reshape(-1,1)
      regressor_r = SVR(kernel = kernel, C=C, epsilon=eps,gamma=gamma,degree=degree)
      regressor_r.fit(X,y)
      self.regressor = regressor_r
      
    elif rv == 'v':
      self.u = self.y - self.regressor.predict(self.X).reshape(-1,1)
      self.u_n = self.y_val - self.y_pred.reshape(-1,1)
      self.u_squared = self.u**2
      self.u_squared_pred = self.u_n**2
      self.df = pd.DataFrame(self.u_squared.append(self.u_squared_pred))

      self.df.rename(columns={'Return':'u_squared'},inplace=True)
      self.df['vol_prox']=(ret_data.iloc[:-1]-ret_data.iloc[:-1].mean())**2
      X,y,X_val,y_val = garch_df(self.df,1,1)
      regressor_v = SVR(kernel = kernel, C=C, epsilon=eps,gamma=gamma,degree=degree)
      regressor_v.fit(X,y)
      self.regressor = regressor_v

    #Predicting a new result
    score = self.regressor.score(X,y)
    y_pred = self.regressor.predict(X_val)
    score_train = self.regressor.score(X,y)
    score_test = self.regressor.score(X_val,y_val)
    self.score = score
    self.y_pred = y_pred
    self.score_train = score_train
    self.score_test = score_test

    if rv=='r':
      rv_name = 'Return'
    elif rv == 'v':
      rv_name = 'Volatility'
    print_msg_box(f'SVR {rv_name}\n -Coin Type: {coins} \n -Period: {period} \n -Kernel: {kernel}')

    #SVR Function
    alpha = self.regressor.dual_coef_
    support_vector = self.regressor.support_vectors_
    if kernel =='poly':
      print('w coef: {}'.format(np.matmul(alpha,(gamma**degree)*(support_vector)**degree)))
      print('b coef: {} \n'.format(self.regressor.intercept_))
      self.w_coef = np.matmul(alpha,(gamma**degree)*(support_vector)**degree)
      self.b_coef = self.regressor.intercept_
    
    elif kernel == 'linear':
      print('w coef: {}'.format(np.matmul(alpha,(support_vector))))
      print('b coef: {} \n'.format(self.regressor.intercept_))
      self.w_coef = np.matmul(alpha,(support_vector))
      self.b_coef = self.regressor.intercept_

    else:
      print('kernel type not supported')
      
    #Model Validation
    mse = mean_squared_error(y_val,y_pred)
    mae = mean_absolute_error(y_val,y_pred)
    rmse = np.sqrt(mean_squared_error(y_val,y_pred))
    aic = calculate_aic(len(X_val),mean_squared_error(y_val,y_pred,),2)
    bic = calculate_bic(len(X_val),mean_squared_error(y_val,y_pred,),2)
    print(f"Train score: {score_train}")
    print(f'Test score: {score_test} \n')
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f'AIC {aic}')
    print(f'BIC {bic} \n')

    if desc_stat_svr:
      print('Descriptive Statistics of Population Data')
      print(stat_desc(y_val),'\n')
      print('Descriptive Statistics of Predicted Data')
      print(stat_desc(y_pred),'\n')

    #Plot
    if plot_svr:
      figure(figsize=(10, 6), dpi=80)
      plt.plot(y_val,label='Observed volatility')
      plt.plot(pd.Series(y_pred, index=y_val.index),'r',ls='--',label='SVR forecasted volatility');
      plt.title('Predicted Volatility')
      plt.xlabel('Days')
      plt.ylabel('Volatility')
      plt.legend(loc='upper right')
      plt.show()
