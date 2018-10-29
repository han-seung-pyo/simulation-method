# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 21:05:37 2018

@author: 한승표
날짜: 2018.08.28~30, 9.3
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
import timeit
# uniform method
def inverse(size):
    uniform = np.random.uniform(size = size)
    norminv = norm.ppf(uniform)
    return uniform, norminv
#Accepntance-Rejection Method
def accepntance(alpha1,alpha2,size):
    beta = (gamma(alpha1)*gamma(alpha2))/ gamma(alpha1+alpha2);
    max_x = (alpha1-1)/(alpha1+alpha2-2)
    c = (max_x**(alpha1-1)*(1-max_x)**(alpha2-1))/beta
    u1 = pd.DataFrame(np.random.uniform(size = (size,1)),columns=['u1'])
    u2 = pd.DataFrame(np.random.uniform(size = (size,1)),columns=['u2'])
    U = pd.concat([u1,u2],axis=1)
    U['fu'] = ((u1**(alpha1-1))*(1-u1)**(alpha2-1))/beta/c
    U['acceptance'] = U.apply(lambda x: x['u1'] if x['fu'] >x['u2'] else 0, axis=1)
    return U


#Box- Muller Method
'''box-muller method algorism
z1, z2 are independent standard normal random numbers
R:반지름, P(R<=x) = 1 -np.e(-x/2)
algorism
1. r = -2 * np.log(U1)
2. V = 2 * np.pi * U2
3. z1 = np.sqrt(r) * np.cos(v), z2 = 2* np.sqrt(r) * np.sin(v)
'''
def box_muller(size):
    data = pd.DataFrame({'R':-2*np.log(np.random.uniform(size = size)),'v' : 2*np.pi * np.random.uniform(size = size)})
    data['z1'] = np.sqrt(data['R']) * np.cos(data['v'])
    data['z2'] = np.sqrt(data['R']) * np.sin(data['v'])
    return data
#Marsaglia and Bray Method(1964)
'''
--> 과거 cos, sin값을 계산하기 힘들어서 개발된 방법론.
Algorism
1. u1 = 2u1 -1
2. u2 = 2u2 -1
3. x = u1^2 +u2^2 <=1 ? accept : reject
4. y = np.sqrt(-2*np.log(x)/x)    /// 여기서 x는 3번에서 accept된 x 이렇게 하면 uniform을 한번더 generate할 필요가 없다. accept된 x가  uniform하므로
5. z1 = u1 * y, z2 = u2 * y
'''
def marsaglia_bray(size):
    data = pd.DataFrame({'U1':2 * np.random.uniform(size = size)-1 , 'U2' :2 * np.random.uniform(size = size) -1 })
    data['x'] = data['U1']**2 + data['U2']**2
    data['y'] = data.apply(lambda row: np.sqrt((np.log(row['x'])*(-2))/row['x']) if row['x'] <=1 else 0, axis=1)
    data['z1'] = data['U1'] * data['y']
    data['z2'] = data['U2'] * data['y']
    return data
size = (1000,1000)
alpha1 = 3;
alpha2 = 2;
size1 = 25000
size2=2500
frequency = np.arange(0.1,1,0.05);
frequency2 = np.arange(-4,5,0.5);
print('-'*50)
print('-'*50)
start = timeit.default_timer()
uniform, norminv = inverse(size)
end = timeit.default_timer()
print('걸린 시간: %2f초'%(end - start))
plt.hist(uniform.flatten(), bins = frequency,rwidth=0.5 ,label = 'Uniform' );
plt.show();
plt.hist(norminv.flatten(), bins=frequency2,  rwidth=0.5, label = 'normal' );
plt.show();
print('-'*50)
print('-'*50)
start = timeit.default_timer()
x = accepntance(alpha1,alpha2,size1)
end = timeit.default_timer()
print('걸린 시간: %2f초'%(end - start))
plt.hist(x['acceptance'], bins = frequency,rwidth=0.5, label = 'Acceptance and Rejection');
plt.show()
count_acceptance = np.count_nonzero(x.iloc[:,-1])
print('count of acceptance is %d'%(count_acceptance))
print('-'*50)
print('-'*50)
start = timeit.default_timer()
bm = box_muller(size2)
end = timeit.default_timer()
print('걸린 시간: %2f초'%(end - start))
plt.hist(bm.ix[:,'z1':'z2'].values.flatten(), bins = frequency2 ,rwidth = 0.5, label = 'Box_Muller Method');
plt.show()
print('-'*50)
print('-'*50)
start = timeit.default_timer()
mb = marsaglia_bray(size1)
end = timeit.default_timer()
print('걸린 시간: %2f초'%(end - start))
plt.hist(mb.ix[:,'z1':'z2'].values.flatten(), bins = frequency2 ,rwidth = 0.5, label='Marsaglia_Bray Method');
plt.show()

