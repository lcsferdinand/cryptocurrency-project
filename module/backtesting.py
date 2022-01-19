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
    
    def generate_data(self,u,y_pred):
        """
        u: predicted u
        """
        self.return_gen=[]
        y_pred = minus_fix(y_pred)
        for i in range(10000):
            self.return_gen.append(u[-1:].values+np.sqrt(y_pred[-1])*random.choice(self.z_hat.reset_index()['Return']))

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
<<<<<<< HEAD
        
        
=======
    
>>>>>>> 81415788b0e7dc71d16077caa2147c9563ae52ca
    def kupiec_test(self, violations, var_conf_level=0.99, conf_level=0.95):
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
        theta= 1-(v/N)
<<<<<<< HEAD

        if v < 0.001:
            V = -2*np.log((1-(v/N))**(N))
        else:
            part1 = ((1-var_conf_level)**(v)) * (var_conf_level**(N-v))
            self.part_1_left =(1-var_conf_level)**(v)
            self.part_1_right = (var_conf_level**(N-v))
            part11= ((1-theta)**(v)) * (theta**(N-v))
            # self.
            
            # fact = math.factorial(N) / ( math.factorial(v) * math.factorial(N-v))
            
            num1 = part1 #* fact
            den1 = part11 #* fact 
        
            V = -2*(np.log(num1/den1))
            self.V_val = V
        
        chi_square_test = chi2.cdf(V,1) #one degree of freedom
        
        if chi_square_test < conf_level: result = "Fail to reject H0"
        elif v==0 and N<=255 and var_conf_level==0.99: result = "Fail to reject H0"
        else: result = "Reject H0"
            
        return {"statictic test":V, "chi square value":chi_square_test, 
                "null hypothesis": f"Probability of failure is {round(1-var_conf_level,3)}",
                "result":result}
=======
>>>>>>> 81415788b0e7dc71d16077caa2147c9563ae52ca

        if v < 0.001:
            V = -2*np.log((1-(v/N))**(N))
        else:
            part1 = ((1-var_conf_level)**(v)) * (var_conf_level**(N-v))
            self.part_1_left =(1-var_conf_level)**(v)
            self.part_1_right = (var_conf_level**(N-v))
            part11= ((1-theta)**(v)) * (theta**(N-v))
            # self.
            
            # fact = math.factorial(N) / ( math.factorial(v) * math.factorial(N-v))
            
            num1 = part1 #* fact
            den1 = part11 #* fact 
        
            V = -2*(np.log(num1/den1))
            self.V_val = V
        
        chi_square_test = chi2.cdf(V,1) #one degree of freedom
        
        if chi_square_test < conf_level: result = "Fail to reject H0"
        elif v==0 and N<=255 and var_conf_level==0.99: result = "Fail to reject H0"
        else: result = "Reject H0"
            
        return {"statictic test":V, "chi square value":chi_square_test, 
                "null hypothesis": f"Probability of failure is {round(1-var_conf_level,3)}",
                "result":result}
