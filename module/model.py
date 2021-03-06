# -*- coding: utf-8 -*-
"""model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XkJBCqK_T7v8gUhjGEGJ3xcT4tQFnA30
"""

class model:
  def model_data(self, ret_data):
    n_test = m.floor(len(ret_data)/4)
    self.ret_data = ret_data
    self.X = ret_data.iloc[:-n_test]
    self.y = ret_data.shift(-1).iloc[:-n_test]
    self.X_val = ret_data.iloc[-n_test:-1]
    self.y_val = ret_data.shift(-1).iloc[-n_test:-1]

  def svr(self, kernel, eps, C=3, gamma=3, degree=3):
    ret_data = self.ret_data
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
    print('Descriptive Statistics of Population Data')
    print(stat_desc(y_val),'\n')
    print('Descriptive Statistics of Predicted Data')
    print(stat_desc(y_pred),'\n')

    #Plot
    figure(figsize=(10, 6), dpi=80)
    plt.plot(y_val,label='Observed volatility');
    plt.plot(pd.Series(y_pred, index=y_val.index),'r',ls='--',label='SVR forecasted volatility');
    plt.title('Predicted Volatility');
    plt.xlabel('Days')
    plt.ylabel('Volatility')
    plt.legend(loc='upper right');
    self.regressor = regressor
    self.y_pred = y_pred

   
    return self#, y_pred, regressor
# period ,coins, ret_data, 
  def svr_based_garch(self, ret_data, kernel, eps, C = 3, gamma = 3, degree = 3):
    # if period == 'normal':
    self.u = self.y - self.regressor.predict(self.X).reshape(-1,1)
    self.u_n = self.y_val - self.y_pred.reshape(-1,1)
    self.u_squared = self.u**2
    self.u_squared_pred = self.u_n**2
    u2 = self.u_squared
    u2_pred = self.u_squared_pred

    self.df_sg = pd.DataFrame(u2.append(u2_pred))
    df_sg = self.df_sg
    df_sg.rename(columns={'Return':'u2'},inplace=True)
    df_sg['vol_prox']=(ret_data.iloc[:-1]-ret_data.iloc[:-1].mean())**2
    X_sg,y_sg,X_sg_val,y_sg_val = garch_df(df_sg,1,1)
    print('X columns: ',X_sg.columns)
    print('y columns: ',y_sg.columns)
    
    y_pred_vol_btc_normal_linear,model_vol_btc = svr_func(X_sg,np.squeeze(y_sg),
                                                        X_sg_val,np.squeeze(y_sg_val),kernel = kernel, C=C, eps=eps,gamma=gamma,degree=degree)
    # globals()['self.%s' % coins,kernel,period]