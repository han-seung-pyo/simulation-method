# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 21:05:37 2018

@author: 한승표
날짜: 2018.08.28~30
과목: 시뮬레이션 방법론
pricing 방법론 중 numerical method에는 3가지가 있다. Lattice(tree model), FDM, monte carlo simuation

random number를 generate하는 방법르로는 uniform, normal, exponential, beta 등의 방법이 존재

1. general purpose method 
-Inverse transform method
-Acceptance-rejection method
위의 방법론은 normal/ exponential/ beta random number을 generate할 수 있다.
자세한 방법론은 lecture note  참조, 혹은 code보면서 이해하기.
2. Special Transform Method : general 방법이 오래 걸릴 때. 
-Box-Muller method
-Marsaglia-Bray method
위의 두가지 방법은 normal random number만 만들어 낼 수 있다.


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import norm
from math import gamma

# uniform method
size = (1000,1000)
frequency = np.arange(0.1,1,0.1);
frequency2 = np.arange(-4,5,1);
def inverse(size):
    uniform = np.random.uniform(size = size)
    norminv = norm.ppf(uniform)
    return uniform, norminv
uniform, norminv = inverse(size)

plt.hist(uniform.flatten(), bins = frequency,rwidth=0.5);
plt.show();
plt.hist(norminv.flatten(), bins=frequency2,  rwidth=0.5);
plt.show();

#Accepntance-Rejection Method
alpha1 = 3;
alpha2 = 2;
size = 10000
def accepntance(alpha1,alpha2,size):
    beta = (gamma(alpha1)*gamma(alpha2))/ gamma(alpha1+alpha2);
    max_x = (alpha1-1)/(alpha1+alpha2-2)
    c = (max_x**(alpha1-1)*(1-max_x)**(alpha2-1))/beta
    u1 = pd.DataFrame(np.random.uniform(size = (size,1)))
    u2 = pd.DataFrame(np.random.uniform(size = (size,1)))
    fu = pd.DataFrame(((u1[0]**(alpha1-1))*(1-u1[0])**(alpha2-1))/beta/c)
    fu['acceptance'] = pd.Series()
    for i in range(len(fu)):
        if fu.iloc[:,0][i] >= u2.iloc[:,0][i]:
            fu['acceptance'][i] = u1[0][i]
        else: 
            fu['acceptance'][i] = 0
    return fu

x = accepntance(alpha1,alpha2,size)
count_acceptance = np.count_nonzero(x.iloc[:,1])

plt.hist(x['acceptance'], bins = frequency,rwidth=0.5);
plt.show()
print('-'*50)
print('count of acceptance is %d'%(count_acceptance))
print('-'*50)
