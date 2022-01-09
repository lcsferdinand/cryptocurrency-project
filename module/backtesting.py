import numpy as np
import math as m
from itertools import chain
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
            q = ((1-r)*np.sort(self.z_hat)[n_f-1])+(r*np.sort(self.z_hat)[n_c-1])
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
            q = ((1-r)*np.sort(self.z_hat)[n_f-1])+(r*np.sort(self.z_hat)[n_c-1])
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

