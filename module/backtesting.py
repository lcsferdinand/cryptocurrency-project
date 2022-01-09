import numpy as np

def minus_fix(minus_array):
  for i in range(len(minus_array)):
    if minus_array[i]<0:
      if i != 0:
        minus_array[i] = minus_array[i-1]
      else:
        minus_array[0] = abs(minus_array[0])
  return minus_array
class risk:

    def var(df_return,u,y_pred,alpha):   
        y_pred =  minus_fix(y_pred)
        self.z_hat = (df_return[u[:-1].index]-u[:-1])/np.sqrt(np.sqrt(y_pred)[:-1])
        var = []
        for i in (alpha):
            n = (len(self.z_hat)+1)*i
            n_f = m.floor(n)
            n_c = m.ceil(n)
            r = n_c-n
            q = ((1-r)*np.sort(self.z_hat)[n_f-1])+(r*np.sort(self.z_hat)[n_c-1])
            var.append(u[-1:].values+np.sqrt(y_pred[-1])*q)
        self.var_mat = np.column_stack((alpha,var))

    def es(df_return,u,y_pred,alpha):
        y_pred = minus_fix(y_pred)
        self.z_hat = (df_return[u[:-1].index]-u[:-1])/np.sqrt(y_pred[:-1])
        es= []
        for i in (alpha):
            n = (len(self.z_hat)+1)*i
            n_f = m.floor(n)
            n_c = m.ceil(n)
            r = n_c-n
            q = ((1-r)*np.sort(self.z_hat)[n_f-1])+(r*np.sort(self.z_hat)[n_c-1])
            es_z = np.mean([x for x in self.z_hat if x < q])
            es.append(u[-1:].values+np.sqrt(y_pred[-1])*es_z)
        self.es_mat = np.column_stack((alpha,es))
    
    def generate_data(u,y_pred):
        self.return_gen=[]
        y_pred = minus_fix(y_pred)
        for i in range(10000):
            self.return_gen.append(u[-1:].values+np.sqrt(y_pred[-1])*random.choice(self.z_hat.reset_index()['Return']))

    def cvar(alpha):
        cvar = []
        for i in (alpha):
            cvar.append(len([x for x in self.return_gen if x < self.var_mat[i][1]])/10000)
        self.cvar_mat = np.column_stack((alpha,cvar))

    def ces(alpha):
        ces = []
        for i in (alpha):
            ces.append(len([x for x in self.return_gen if x < self.var_es[i][1]])/10000)
        self.cvar_es = np.column_stack((alpha,ces))

