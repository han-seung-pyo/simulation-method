# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 05:21:57 2019

@author: 한승표
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

start_time = time.time() 
#------------------------
s0 = 100
k = 105
T= 1
r = 0.05
sigma = 0.2
n = 50000
n_steps = 100
z = np.random.standard_normal(n)
s_T = s0 * np.exp((r-0.5*sigma**2)*T + sigma*np.sqrt(T)*z)
q=0
#montecarlo simulation
def MontecarloSim(s0,k,r,T,sigma,q,n,n_steps):  
    delta_t =T/n_steps
    z_matrix = np.random.standard_normal(size =(n,(n_steps)))
    st_matrix = np.zeros((n,n_steps))
    st_matrix[:,0] = s0
    for i in range(n_steps-1):
        st_matrix[:,i+1] = st_matrix[:,i]*np.exp((r-q-0.5*sigma**2)*delta_t + sigma*np.sqrt(delta_t)*z_matrix[:,i])
    return st_matrix

MontecarloSim(s0,k,r,T,sigma,q,n,n_steps)
#----------------------------
#종료부분 코드
#print("start_time", start_time) #출력해보면, 시간형식이 사람이 읽기 힘든 일련번호형식입니다.#
print("--- %s seconds ---" %(time.time() - start_time))