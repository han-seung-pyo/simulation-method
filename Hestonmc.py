# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:10:14 2018

@author: 한승표
"""

import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import norm
import matplotlib.pyplot as plt


def montecarlo(s,k,r,q,t,sigma,option_type,M):    
    data = pd.DataFrame(norm.ppf(np.random.uniform(size = M)), columns = ['epsilon'])
    data['sT'] = data.apply(lambda x : s * np.exp((r-q-0.5*sigma**2)*t + sigma * np.sqrt(t) * x['epsilon']),axis =1)
    data['option_price'] = data.apply(lambda x: max(x['sT']-k,0) if option_type =='call' else max(k-x['sT'],0), axis = 1)
    return data, data['option_price'].sum()/M * np.exp(-r*t).round(4)


def hestonmc(S0, K, T, r, q, sigma, kappa,lam, theta, rho, V0, n_trials, n_steps,boundaryScheme):
    dt = T/n_steps
    kappa_new = kappa+ lam
    theta_new = (kappa*theta)/(kappa+lam)
    
    randn_matrix_1 = np.random.normal(size=(n_trials, n_steps)) #epsilon1
    randn_matrix_2 = np.random.normal(size=(n_trials, n_steps)) #epsilon2
    randn_matrix_S = randn_matrix_1
    randn_matrix_v =  rho * randn_matrix_S + np.sqrt(1 - rho ** 2) * randn_matrix_2

    # boundary scheme fuctions
    if (boundaryScheme == "absorption"):
        f1 = f2 = f3 = lambda x: np.maximum(x, 0)
    elif (boundaryScheme == "reflection"):
        f1 = f2 = f3 = np.absolute
    elif (boundaryScheme == "Higham and Mao"):
        f1 = f2 = lambda x: x
        f3 = np.absolute
    elif (boundaryScheme == "partial truncation"):
        f1 = f2 = lambda x: x
        f3 = lambda x: np.maximum(x, 0)
    elif (boundaryScheme == "full truncation"):
        f1 = lambda x: x
        f2 = f3 = lambda x: np.maximum(x, 0)
    V_matrix = np.zeros((n_trials, n_steps + 1))
    V_matrix[:, 0] = V0
    log_price_matrix = np.zeros((n_trials, n_steps + 1))
    log_price_matrix[:, 0] = np.log(S0)    
    for j in range(n_steps):
        V_matrix[:, j + 1] = f1(V_matrix[:, j]) + kappa_new  * (theta_new - f2(V_matrix[:, j])) *dt + \
                                 sigma * np.sqrt(f3(V_matrix[:, j])) * np.sqrt(dt) * randn_matrix_v[:, j] #V 틸다 메트릭스
        V_matrix[:, j + 1] = f3(V_matrix[:, j + 1]) # lnS process에 들어갈 값
        log_price_matrix[:, j + 1] = log_price_matrix[:, j] + (r-q - 0.5*f2(V_matrix[:, j])) * dt + \
                                         np.sqrt(f3(V_matrix[:, j])) * np.sqrt(dt) * randn_matrix_S[:, j]
    price_matrix = np.exp(log_price_matrix)
    callprice_matrix = np.maximum(price_matrix[:,-1]-k,0) * np.exp(-r *n_steps /365)
    call_price = callprice_matrix.sum() / n_trials
    err  = np.std(callprice_matrix) /np.sqrt(n_trials)
    return call_price,err


S0=100
k=100
t=30/365
r= 0
q = 0
V0=0.01
kappa=2
theta=0.01
lam = 0
sigma = 0.1
rho=0
n_steps = 100
n_trials = 1000
boundaryScheme = "full truncation" 
price = np.ones(10)
err = np.ones(10)
for i in range(10):
    n_trials = 1000 *(i+1)
    price[i] = (hestonmc(S0,k,t,r,q,sigma,kappa,lam,theta,rho,V0,n_trials,n_steps,boundaryScheme)[0])
    err[i] = (hestonmc(S0,k,t,r,q,sigma,kappa,lam,theta,rho,V0,n_trials,n_steps,boundaryScheme)[1])
c1 =  price - 1.96*err
c2 = price+1.96*err
call_price = 1.1345
plt.plot(c2)
plt.plot(c1)
#plt.plot(call_price)
plt.legend()
#plt.xlim([1000,10000])
plt.show()

print("Heston call_price: %4f"  %(1.1345))
print("bs_model call price: %4f" %(bs_price(S0,k,r,q,t,sigma,'call')))
print("monte carlo call price: %4f :" %(montecarlo(S0,k,r,q,t,sigma,'call',2000)[1]))
print("heston monte carol call price %4f" %(hestonmc(S0,k,t,r,q,sigma,kappa,lam,theta,rho,V0,n_trials,n_steps,boundaryScheme)[0]))


#            