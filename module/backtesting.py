import numpy as np
import math as m
from itertools import chain
import random
import pandas as pd
# import vartests
from scipy import stats
import numpy as np
from scipy import optimize
from scipy.stats import chi2
import math
import arch
import time
import pandas as pd
import random

def minus_fix(minus_array):
  for i in range(len(minus_array)):
    if minus_array[i]<0:
      if i != 0:
        minus_array[i] = minus_array[i-1]
      else:
        minus_array[0] = abs(minus_array[0])
  return minus_array
  
class risk:

    def var(self,df_return,u,y_pred,alpha):
        """
        u: predicted u
        """
        alpha = np.array(alpha).reshape(len(alpha),1)   
        y_pred =  minus_fix(y_pred)
        self.z_hat = (df_return.loc[u[:-1].index]-u[:-1])/np.sqrt([y_pred[:-1]]).T
        var = []
        j = 0
        for i in (alpha):
            n = (len(self.z_hat)+1)*i
            n_f = m.floor(n)
            n_c = m.ceil(n)
            r = n_c-n
            q = ((1-r)*np.sort(np.array(self.z_hat).flatten())[n_f-1])+(r*np.sort(np.array(self.z_hat).flatten())[n_c-1])
            var.append(u[-1:].values+np.sqrt(y_pred[-1])*q)
            var[j] = list(chain(*var[j]))
            j+=1
        self.var_mat = np.column_stack((alpha,var))

    def es(self,df_return,u,y_pred,alpha):
        """
        u: predicted u
        """
        alpha = np.array(alpha).reshape(len(alpha),1) 
        y_pred = minus_fix(y_pred)
        self.z_hat = (df_return.loc[u[:-1].index]-u[:-1])/np.sqrt([y_pred[:-1]]).T
        es= []
        j=0
        for i in (alpha):
            n = (len(self.z_hat)+1)*i
            n_f = m.floor(n)
            n_c = m.ceil(n)
            r = n_c-n
            q = ((1-r)*np.sort(np.array(self.z_hat).flatten())[n_f-1])+(r*np.sort(np.array(self.z_hat).flatten())[n_c-1])
            es_z = np.mean([x for x in self.z_hat[self.z_hat.columns[0]] if x < q])
            es.append(u[-1:].values+np.sqrt(y_pred[-1])*es_z)
            es[j] = list(chain(*es[j]))
            j+=1
        self.es_mat = np.column_stack((alpha,es))
    
    def generate_data(self,u,y_pred,seed=0):
        """
        u: predicted u
        """
        self.return_gen=[]
        y_pred = minus_fix(y_pred)
        random.seed(seed)
        for i in range(10000):
            self.return_gen.append(u[-1:].iloc[0][0]+np.sqrt(y_pred[-1])*random.choice(self.z_hat.reset_index()['Return']))

    def cvar(self,alpha):
        cvar = []
        alpha = np.array(alpha).reshape(len(alpha),1)   
        for i in range(len(alpha)):
            cvar.append(len([x for x in self.return_gen if x < self.var_mat[i][1]])/10000)
        self.cvar_mat = np.column_stack((alpha,cvar))

    def ces(self,alpha):
        ces = []
        alpha = np.array(alpha).reshape(len(alpha),1)   
        for i in range(len(alpha)):
            ces.append(len([x for x in self.return_gen if x < self.es_mat[i][1]])/10000)
        self.ces_mat = np.column_stack((alpha,ces))
        

    def kupiec_test(self, violations, var_conf_level=0.99, conf_level=0.05):
        '''Perform Kupiec Test (1995).
        The main goal is to verify if the number of violations, i.e. proportion of failures, is consistent with the
        violations predicted by the model.
        
            Parameters:
                violations (series):    series of violations of VaR
                var_conf_level (float): VaR confidence level
                conf_level (float):     test confidence level
            Returns:
                answer (dict):          statistics and decision of the test
        '''
        if isinstance(violations, pd.core.series.Series):
            v = violations[violations==1].count()
        elif isinstance(violations, pd.core.frame.DataFrame):
            v = violations[violations==1].count().values[0]

        N = violations.shape[0]
        theta= 1-(v/N) #return higher than VaR proportion 

        if v < 0.001:
            V = -2*np.log((1-(v/N))**(N))
        else:
            
            left = ((var_conf_level/theta)**(N-v))
            right = ((1-var_conf_level)/(1-theta))**(v)
        
            V = -2*(np.log(left*right))
            self.V_val = V
        
        p_val = 1-chi2.cdf(V,1) #one degree of freedom
        
        if p_val < conf_level: result = "Fail to reject H0"
        # elif v==0 and N<=255 and var_conf_level==0.99: result = "Fail to reject H0"
        else: result = "Reject H0"
            
        return {"Chi Square Stat Value":V, "P Value":p_val, 
                "null hypothesis": f"Probability of failure is {round(1-var_conf_level,3)}",
                "result":result}
                
    def es_backtesting(self,risk,i,K=1000,conf_level = 0.05):
      # generate y_t series
      ES = risk[1][i]
      x_t = [x for x in risks_btc_garch_n.return_gen if x < ES]
      vol = minus_fix(m_btc_garch_n.y_pred)[-1]
      y_t = (x_t-ES)/np.sqrt(vol)
      var_conf_level = risks_btc_garch_n.es_mat[0][i]

      #generate I_t
      I_t = y_t - y_t.mean()

      #bootstrap I_t
      I_t_dict = defaultdict(list)
      I_t_dict['I_t_0'].extend(I_t)

      for i in range(1,K+1):
        random.seed(i)
        I_t_arr=[]
        for j in range(len(I_t)):
          I_t_arr.append(random.choice(I_t))
        I_t_dict['I_t_'+str(i)].extend(I_t_arr)

      #generate t(I)
      t_I = []
      for i in range(K+1):
        mean = np.mean(I_t_dict['I_t_'+str(i)])
        std = np.std(I_t_dict['I_t_'+str(i)])
        t_I.append(mean/std)

      #p-val
      p_val = np.sum([1 for x in t_I[1:] if x > t_I[0]])/K

      if p_val < conf_level: result = 'Fail to reject H0'
      else: result = 'Reject H0'

      return {"P Value":p_val, 
              "null hypothesis": f"Probability of failure is {round(var_conf_level,3)}",
              "result":result}
