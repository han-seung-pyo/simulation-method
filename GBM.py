# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 15:49:58 2018

@author: 한승표
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
mu = 0.1;
vol = 0.2;
dt = 1/(250);
init  = 2285;
size = 250;

def bgm(mu,vol,dt,init, size):
    norminv = pd.DataFrame(norm.ppf(np.random.uniform(size = size)), columns = ['epsilon'])
    norminv['gbm'] = pd.Series()
    for i in range(len(norminv)+1):
        if i == 0 :
            norminv['gbm'][0] = init
        else:
           norminv['gbm'][i] = norminv['gbm'][i-1] * np.exp((mu -0.5 *vol**2)*dt + vol*np.sqrt(dt)*norminv['epsilon'][i-1])
    return norminv

bgm_val = bgm(mu,vol,dt,init,size)
plt.plot(bgm_val['gbm'])

#%%

def montecarlo(s,k,r,q,t,sigma,option_type,M):    
    data = pd.DataFrame(norm.ppf(np.random.uniform(size = M)), columns = ['epsilon'])
    data['sT'] = data.apply(lambda x : s * np.exp((r-q-0.5*sigma**2)*t + sigma * np.sqrt(t) * x['epsilon']),axis =1)
    data['option_price'] = data.apply(lambda x: max(x['sT']-k,0) if option_type =='call' else max(k-x['sT'],0), axis = 1)
    return data['option_price'].sum()/M * np.exp(-r*t)
    
    
def stratifed(s,k,r,q,t,sigma,option_tpye,M):
    x = 0
    for i in range(M):
        epsilon = np.nan_to_num(norm.ppf((i-0.5)/M))
        s_T = s * np.exp((r-q-0.5*sigma**2)*t + sigma * np.sqrt(t) * epsilon)
        if option_tpye == 'call':
            xx = max(s_T-k,0)
        else:
            xx = max(k-s_T,0)
        x = x+ xx
        stratifed = (x / M) *np.exp(-r*t)
    return stratifed

s = 217
k =215
r = 0.05
q = 0
t = 1/12
vol = 0.3
option_type = 'put'

option_value = montecarlo(s,k,r,q,t,vol,option_type,100000).round(4)
