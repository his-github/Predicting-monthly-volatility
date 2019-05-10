#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 20:55:40 2019

@author: mu
"""




import numpy as np

import pandas as pd

from pandas import Series, DataFrame

import matplotlib.pyplot as plt

from data_clean import Stock

def loss(real_v,model_v):
        # Root Mean Square Error
        RMSE=np.sqrt(np.nanmean(np.square(real_v-model_v)))
        #Mean Absolute Percent Error
        MAPE=np.nanmean(np.abs(real_v-model_v)/real_v)
        #Mean absolute Error
        MAE=np.nanmean(np.abs(real_v-model_v))
        return DataFrame([RMSE,MAPE,MAE],index=["RMSE","MAPE","MAE"],columns=['loss'])
def a(n,x,derivative=False):
    a=np.matrix(np.array([x**i for i in reversed(range(n))]))
    b=np.matrix(np.array([i*x**(i-1) for i in reversed(range(n))]))
    if derivative:
        return a*(n*x**(n-1)+(1-n)*x**n-1)/(1-x**n)**2+b*(1-x)/(1-x**n)
    else:
        return a*(1-x)/(1-x**n)
    
def fit(X,target,hold_out,d,rate=0.001,beta=0.9,error=10**-8,iteration=10**4,observe=False):
    """x is the trainning set, Y is monthly volatility,d is the day lag used
    use gradient descent to minimize mean square loss
    both X,Y are numpy.ndarray
    """
    Y=np.array(target[d+21:])
    
    np.random.seed()
    mu=np.random.uniform(0.1,1)
    A=np.matrix(np.zeros((d,d)))
    v2=np.matrix(np.zeros(d))
    
    N=hold_out
    for i in range(N):
        v2+=Y[i]*X[i:i+d]
        A+=np.matrix(X[i:i+d]).T*np.matrix(X[i:i+d])
    step=0
    g_mu=0.1
    
    update_mu=0
    while step<iteration and g_mu>error:
        g_mu=np.asscalar((-v2+a(d,mu)*A)*a(d,mu,True).T)
        update_mu=update_mu*beta+g_mu*(1-beta)
        mu-=rate*update_mu
        step+=1
        if step%100==0 and observe:
            L=0
            for i in range(N):
                L+=(Y[i]-np.asscalar(a(d,mu)*np.matrix(X[i:i+d]).T))**2
            print("mu,gradient,loss",mu,g_mu,np.sqrt(L/N))
    Loss_in_sample,Loss_out_of_sample=0,0
    In_sample_fit=np.zeros(N)
    for i in range(N):
        In_sample_fit[i]=np.asscalar((a(d,mu)*np.matrix(X[i:i+d]).T))
    Loss_in_sample=np.sum(np.square(Y[:N]-In_sample_fit))
    RMSE_in_sample=np.sqrt(Loss_in_sample/N)
    print("in smaple",RMSE_in_sample)
    plt.plot(In_sample_fit)
    num_hold_out=len(Y)-N
    out_of_sample_fit=np.zeros(len(target))
    for i in range(N,N+num_hold_out):
        out_of_sample_fit[i]=np.asscalar((a(d,mu)*np.matrix(X[i:i+d]).T))
        Loss_out_of_sample +=(Y[i]-out_of_sample_fit[i])**2
    RMSE_out_of_sample=np.sqrt(Loss_out_of_sample/num_hold_out)
    plt.plot(out_of_sample_fit)
    plt.plot(Y)
    print("out of smaple", RMSE_out_of_sample)
    return mu
        
    
    

stockprice=pd.read_csv('stockdata2.csv')
stockprice.loc[:,'a':].plot(title='Before data clean',subplots=True,layout=(3,2),figsize=(12,18))
stk=Stock(stockprice)
stk.clean()
stk.price.loc[:,'a':].plot(title='After data clean, before deepclean',subplots=True,layout=(3,2),figsize=(12,18))

stk.deepclean(['a','b','c','d','e','f'])

stk.price.loc[:,'a':].plot(title='After deepclean',subplots=True,layout=(3,2),figsize=(12,18))

stk.split_adjust('c')
plt.plot(stk.price.c)
