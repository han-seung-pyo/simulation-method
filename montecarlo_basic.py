# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 04:25:08 2019

@author: 한승표
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
#page40
s = 100
k = 105
T= 1
r = 0.05
sigma = 0.2
sim_num = 1000
n_steps = 100

#%%
def Montecarlo_sim(s,T,r,q,sigma,sim_num,n_steps):
    delta_t = T/n_steps
    z_matrix = np.random.standard_normal(size =(sim_num,n_steps))
    st_matrix = np.zeros((sim_num,n_steps))
    st_matrix[:,0] = s
    for i in range(n_steps-1):
        st_matrix[:,i+1] = st_matrix[:,i]*np.exp((r-q-0.5*sigma**2)*delta_t + sigma*np.sqrt(delta_t)*z_matrix[:,i])
    
    return st_matrix

S_matrix = Montecarlo_sim(s,T,r,sigma,sim_num,n_steps)
