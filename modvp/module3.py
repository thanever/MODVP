# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:13:50 2016

@author: T.Han
"""

import sys
reload(sys)
sys.setdefaultencoding('utf8')

from module1 import  FormFile, FormCard, Call, CalculateIndex
import numpy as np
from sklearn.svm import SVR
from sklearn import preprocessing
from pandas import read_csv, DataFrame
from itertools import combinations
import random

class Regis:
    """
    ||生成训练样本及样本数据解析
    """
    @staticmethod
    def sample_point(bus_potential, gap, max_rate):
        """
        ||max_rate - 补偿装置的最大容量
        ||gap - 样本点中容量间隔
        """
        n_potential_bus = len(bus_potential)
        X = list()
        X.append([0]*2*n_potential_bus)
        '''
        for i in range(n_potential_bus):
            for j in range(gap['svc'], max_rate['svc']+gap['svc'], gap['svc']):
                l = [0]*2*n_potential_bus
                l[2*i] = j
                X.append(l)
            for j in range(gap['sta'], max_rate['sta']+gap['sta'], gap['sta']):
                l = [0]*2*n_potential_bus
                l[2*i+1] = j
                X.append(l)
        for (i,j) in combinations(range(n_potential_bus),2):
            for k1 in range(gap['svc'], max_rate['svc']+gap['svc'], gap['svc']):
                for k2 in range (gap['svc'], max_rate['svc']+gap['svc'], gap['svc']):
                    l = [0]*2*n_potential_bus
                    l[2*i] = k1
                    l[2*j] = k2
                    X.append(l)
            for k1 in range(gap['sta'], max_rate['sta']+gap['sta'], gap['sta']):
                for k2 in range (gap['sta'], max_rate['sta']+gap['sta'], gap['sta']):
                    l = [0]*2*n_potential_bus
                    l[2*i+1] = k1
                    l[2*j+1] = k2
                    X.append(l)
            for k1 in range(gap['svc'], max_rate['svc']+gap['svc'], gap['svc']):
                for k2 in range (gap['sta'], max_rate['sta']+gap['sta'], gap['sta']):
                    l = [0]*2*n_potential_bus
                    l[2*i] = k1
                    l[2*j+1] = k2
                    X.append(l)
            for k1 in range(gap['sta'], max_rate['sta']+gap['sta'], gap['sta']):
                for k2 in range (gap['svc'], max_rate['svc']+gap['svc'], gap['svc']):
                    l = [0]*2*n_potential_bus
                    l[2*i+1] = k1
                    l[2*j] = k2
                    X.append(l)
        for (i,j,k) in combinations(range(n_potential_bus),3):
            l = [0]*2*n_potential_bus
            l[2*i] = gap['svc']
            l[2*j] = gap['svc']
            l[2*k] = gap['svc']
            X.append(l)
            l = [0]*2*n_potential_bus
            l[2*i] = gap['svc']
            l[2*j+1] = gap['sta']
            l[2*k] = gap['svc']
            X.append(l)
            l = [0]*2*n_potential_bus
            l[2*i] = gap['sta']
            l[2*j+1] = max_rate['svc']
            l[2*k] = gap['sta']
            X.append(l)
        
        for (i,j,k) in combinations(range(n_potential_bus),3):
            l = [0]*2*n_potential_bus
            l[2*i+1] = gap['sta']
            l[2*j] = gap['svc']
            l[2*k+1] = gap['sta']
            X.append(l)  
            l = [0]*2*n_potential_bus
            l[2*i+1] = gap['sta']
            l[2*j+1] = gap['sta']
            l[2*k+1] = gap['sta']
            X.append(l)  
        for (i,j,k,r) in combinations(range(n_potential_bus),4):
            l = [0]*2*n_potential_bus
            l[2*i] = gap['svc']
            l[2*j] = gap['svc']
            l[2*k] = gap['svc']
            l[2*r] = gap['svc']
            X.append(l)
            l = [0]*2*n_potential_bus
            l[2*i+1] = gap['sta']
            l[2*j+1] = gap['sta']
            l[2*k+1] = gap['sta']
            l[2*r+1] = gap['sta']
            X.append(l) 
            l = [0]*2*n_potential_bus
            l[2*i] = gap['svc']
            l[2*j] = gap['svc']
            l[2*k+1] = gap['sta']
            l[2*r+1] = gap['sta']
            X.append(l)
        
        for (i,j,k,r,t) in combinations(range(n_potential_bus),5):
            l = [0]*2*n_potential_bus
            l[2*i] = gap['svc']
            l[2*j] = gap['svc']
            l[2*k] = gap['svc']
            l[2*r] = gap['svc']
            l[2*t] = gap['svc']
            X.append(l)
            l = [0]*2*n_potential_bus
            l[2*i+1] = gap['sta']
            l[2*j+1] = gap['sta']
            l[2*k+1] = gap['sta']
            l[2*r+1] = gap['sta']
            l[2*t+1] = gap['sta']
            X.append(l) 
            l = [0]*2*n_potential_bus
            l[2*i] = gap['svc']
            l[2*j] = gap['svc']
            l[2*k+1] = gap['sta']
            l[2*r+1] = gap['sta']
            l[2*t+1] = gap['sta']
            X.append(l)
        for (i,j,k,r,t,f) in combinations(range(n_potential_bus),6):
            l = [0]*2*n_potential_bus
            l[2*i] = gap['svc']
            l[2*j] = gap['svc']
            l[2*k] = gap['svc']
            l[2*r] = gap['svc']
            l[2*t] = gap['svc']
            l[2*f] = gap['svc']
            X.append(l)
            l = [0]*2*n_potential_bus
            l[2*i+1] = gap['sta']
            l[2*j+1] = gap['sta']
            l[2*k+1] = gap['sta']
            l[2*r+1] = gap['sta']
            l[2*t+1] = gap['sta']
            l[2*f+1] = gap['sta']
            X.append(l) 
            l = [0]*2*n_potential_bus
            l[2*i] = gap['svc']
            l[2*j] = gap['svc']
            l[2*f] = gap['svc']
            l[2*k+1] = gap['sta']
            l[2*r+1] = gap['sta']
            l[2*t+1] = gap['sta']
            X.append(l)
  
        for (i,j,k,r,t,f,w) in combinations(range(n_potential_bus),7):
            l = [0]*2*n_potential_bus
            l[2*i] = gap['svc']
            l[2*j] = gap['svc']
            l[2*k] = gap['svc']
            l[2*r] = gap['svc']
            l[2*t] = gap['svc']
            l[2*f] = gap['svc']
            l[2*w] = gap['svc']
            X.append(l)
            l = [0]*2*n_potential_bus
            l[2*i+1] = gap['sta']
            l[2*j+1] = gap['sta']
            l[2*k+1] = gap['sta']
            l[2*r+1] = gap['sta']
            l[2*t+1] = gap['sta']
            l[2*f+1] = gap['sta']
            l[2*w+1] = gap['sta']
            X.append(l) 
            l = [0]*2*n_potential_bus
            l[2*i] = gap['svc']
            l[2*j] = gap['svc']
            l[2*f] = gap['svc']
            l[2*k+1] = gap['sta']
            l[2*r+1] = gap['sta']
            l[2*t+1] = gap['sta']
            l[2*w+1] = gap['sta']
            X.append(l)
            l = [0]*2*n_potential_bus
            l[2*i+1] = gap['sta']
            l[2*j+1] = gap['sta']
            l[2*f+1] = gap['sta']
            l[2*k] = gap['svc']
            l[2*r] = gap['svc']
            l[2*t] = gap['svc']
            l[2*w] = gap['svc']
            X.append(l)
        for (i,j,k,r,t,f,w) in combinations(range(n_potential_bus),7):
            l = [0]*2*n_potential_bus
            l[2*i] = 2*gap['svc']
            l[2*j] = 2*gap['svc']
            l[2*k] = 2*gap['svc']
            l[2*r] = 2*gap['svc']
            l[2*t] = 2*gap['svc']
            l[2*f] = 2*gap['svc']
            l[2*w] = 2*gap['svc']
            X.append(l)
            l = [0]*2*n_potential_bus
            l[2*i+1] = 2*gap['sta']
            l[2*j+1] = 2*gap['sta']
            l[2*k+1] = 2*gap['sta']
            l[2*r+1] = 2*gap['sta']
            l[2*t+1] = 2*gap['sta']
            l[2*f+1] = 2*gap['sta']
            l[2*w+1] = 2*gap['sta']
            X.append(l) 
            l = [0]*2*n_potential_bus
            l[2*i] = 2*gap['svc']
            l[2*j] = 2*gap['svc']
            l[2*f] = 2*gap['svc']
            l[2*k+1] = 2*gap['sta']
            l[2*r+1] = 2*gap['sta']
            l[2*t+1] = 2*gap['sta']
            l[2*w+1] = 2*gap['sta']
            X.append(l)
            l = [0]*2*n_potential_bus
            l[2*i+1] = 2*gap['sta']
            l[2*j+1] = 2*gap['sta']
            l[2*f+1] = 2*gap['sta']
            l[2*k] = 2*gap['svc']
            l[2*r] = 2*gap['svc']
            l[2*t] = 2*gap['svc']
            l[2*w] = 2*gap['svc']
            X.append(l)
            
        for (i,j,k,r,t,f,w,v) in combinations(range(n_potential_bus),8):
            l = [0]*2*n_potential_bus
            l[2*i] = 2*gap['svc']
            l[2*j] = 2*gap['svc']
            l[2*k] = 2*gap['svc']
            l[2*r] = 2*gap['svc']
            l[2*t] = 2*gap['svc']
            l[2*f] = 2*gap['svc']
            l[2*w] = 2*gap['svc']
            l[2*v] = 2*gap['svc']
            X.append(l)
            l = [0]*2*n_potential_bus
            l[2*i+1] = 2*gap['sta']
            l[2*j+1] = 2*gap['sta']
            l[2*k+1] = 2*gap['sta']
            l[2*r+1] = 2*gap['sta']
            l[2*t+1] = 2*gap['sta']
            l[2*f+1] = 2*gap['sta']
            l[2*w+1] = 2*gap['sta']
            l[2*v+1] = 2*gap['sta']
            X.append(l) 
            l = [0]*2*n_potential_bus
            l[2*i] = 2*gap['svc']
            l[2*j] = 2*gap['svc']
            l[2*f] = 2*gap['svc']
            l[2*k+1] = 2*gap['sta']
            l[2*r+1] = 2*gap['sta']
            l[2*t+1] = 2*gap['sta']
            l[2*w+1] = 2*gap['sta']
            l[2*v+1] = 2*gap['sta']
            X.append(l)
            l = [0]*2*n_potential_bus
            l[2*i+1] = 2*gap['sta']
            l[2*j+1] = 2*gap['sta']
            l[2*f+1] = 2*gap['sta']
            l[2*k] = 2*gap['svc']
            l[2*r] = 2*gap['svc']
            l[2*t] = 2*gap['svc']
            l[2*w] = 2*gap['svc']
            l[2*v] = 2*gap['svc']
            X.append(l)
      

        for (i,j,k,r,t,f,w,v) in combinations(range(n_potential_bus),8):
            l = [0]*2*n_potential_bus
            l[2*i] = gap['svc']
            l[2*j] = gap['svc']
            l[2*k] = gap['svc']
            l[2*r] = gap['svc']
            l[2*t] = gap['svc']
            l[2*f] = gap['svc']
            l[2*w] = gap['svc']
            l[2*v] = gap['svc']
            X.append(l)
            l = [0]*2*n_potential_bus
            l[2*i+1] = gap['sta']
            l[2*j+1] = gap['sta']
            l[2*k+1] = gap['sta']
            l[2*r+1] = gap['sta']
            l[2*t+1] = gap['sta']
            l[2*f+1] = gap['sta']
            l[2*w+1] = gap['sta']
            l[2*v+1] = gap['sta']
            X.append(l) 
            l = [0]*2*n_potential_bus
            l[2*i] = gap['svc']
            l[2*j] = gap['svc']
            l[2*f] = gap['svc']
            l[2*k+1] = gap['sta']
            l[2*r+1] = gap['sta']
            l[2*t+1] = gap['sta']
            l[2*w+1] = gap['sta']
            l[2*v+1] = gap['sta']
            X.append(l)
            l = [0]*2*n_potential_bus
            l[2*i+1] = gap['sta']
            l[2*j+1] = gap['sta']
            l[2*f+1] = gap['sta']
            l[2*k] = gap['svc']
            l[2*r] = gap['svc']
            l[2*t] = gap['svc']
            l[2*w] = gap['svc']
            l[2*v] = gap['svc']
            X.append(l)
            
        
        for (i,j,k,r,t,f,w) in combinations(range(n_potential_bus),7):
            l = [0]*2*n_potential_bus
            l[2*i] = 0.5*gap['svc']
            l[2*j] = 0.5*gap['svc']
            l[2*k] = 0.5*gap['svc']
            l[2*r] = 0.5*gap['svc']
            l[2*t] = 0.5*gap['svc']
            l[2*f] = 0.5*gap['svc']
            l[2*w] = 0.5*gap['svc']
            X.append(l)
            l = [0]*2*n_potential_bus
            l[2*i+1] = 0.5*gap['sta']
            l[2*j+1] = 0.5*gap['sta']
            l[2*k+1] = 0.5*gap['sta']
            l[2*r+1] = 0.5*gap['sta']
            l[2*t+1] = 0.5*gap['sta']
            l[2*f+1] = 0.5*gap['sta']
            l[2*w+1] = 0.5*gap['sta']
            X.append(l) 
            l = [0]*2*n_potential_bus
            l[2*i] = 0.5*gap['svc']
            l[2*j] = 0.5*gap['svc']
            l[2*f] = 0.5*gap['svc']
            l[2*k+1] = 0.5*gap['sta']
            l[2*r+1] = 0.5*gap['sta']
            l[2*t+1] = 0.5*gap['sta']
            l[2*w+1] = 0.5*gap['sta']
            X.append(l)
            l = [0]*2*n_potential_bus
            l[2*i+1] = 0.5*gap['sta']
            l[2*j+1] = 0.5*gap['sta']
            l[2*f+1] = 0.5*gap['sta']
            l[2*k] = 0.5*gap['svc']
            l[2*r] = 0.5*gap['svc']
            l[2*t] = 0.5*gap['svc']
            l[2*w] = 0.5*gap['svc']
            X.append(l)
        for (i,j,k,r,t,f,w) in combinations(range(n_potential_bus),7):
            l = [0]*2*n_potential_bus
            l[2*i] = 4*gap['svc']
            l[2*j] = 4*gap['svc']
            l[2*k] = 4*gap['svc']
            l[2*r] = 4*gap['svc']
            l[2*t] = 4*gap['svc']
            l[2*f] = 4*gap['svc']
            l[2*w] = 4*gap['svc']
            X.append(l)
            l = [0]*2*n_potential_bus
            l[2*i+1] = 4*gap['sta']
            l[2*j+1] = 4*gap['sta']
            l[2*k+1] = 4*gap['sta']
            l[2*r+1] = 4*gap['sta']
            l[2*t+1] = 4*gap['sta']
            l[2*f+1] = 4*gap['sta']
            l[2*w+1] = 4*gap['sta']
            X.append(l) 
            l = [0]*2*n_potential_bus
            l[2*i] = 4*gap['svc']
            l[2*j] = 4*gap['svc']
            l[2*f] = 4*gap['svc']
            l[2*k+1] = 4*gap['sta']
            l[2*r+1] = 4*gap['sta']
            l[2*t+1] = 4*gap['sta']
            l[2*w+1] = 4*gap['sta']
            X.append(l)
            l = [0]*2*n_potential_bus
            l[2*i+1] = 4*gap['sta']
            l[2*j+1] = 4*gap['sta']
            l[2*f+1] = 4*gap['sta']
            l[2*k] = 4*gap['svc']
            l[2*r] = 4*gap['svc']
            l[2*t] = 4*gap['svc']
            l[2*w] = 4*gap['svc']
            X.append(l)
            
        for (i,j,k,r,t,f,w,v) in combinations(range(n_potential_bus),8):
            l = [0]*2*n_potential_bus
            l[2*i] = 4*gap['svc']
            l[2*j] = 4*gap['svc']
            l[2*k] = 4*gap['svc']
            l[2*r] = 4*gap['svc']
            l[2*t] = 4*gap['svc']
            l[2*f] = 4*gap['svc']
            l[2*w] = 4*gap['svc']
            l[2*v] = 4*gap['svc']
            X.append(l)
            l = [0]*2*n_potential_bus
            l[2*i+1] = 4*gap['sta']
            l[2*j+1] = 4*gap['sta']
            l[2*k+1] = 4*gap['sta']
            l[2*r+1] = 4*gap['sta']
            l[2*t+1] = 4*gap['sta']
            l[2*f+1] = 4*gap['sta']
            l[2*w+1] = 4*gap['sta']
            l[2*v+1] = 4*gap['sta']
            X.append(l) 
            l = [0]*2*n_potential_bus
            l[2*i] = 4*gap['svc']
            l[2*j] = 4*gap['svc']
            l[2*f] = 4*gap['svc']
            l[2*k+1] = 4*gap['sta']
            l[2*r+1] = 4*gap['sta']
            l[2*t+1] = 4*gap['sta']
            l[2*w+1] = 4*gap['sta']
            l[2*v+1] = 4*gap['sta']
            X.append(l)
            l = [0]*2*n_potential_bus
            l[2*i+1] = 4*gap['sta']
            l[2*j+1] = 4*gap['sta']
            l[2*f+1] = 4*gap['sta']
            l[2*k] = 4*gap['svc']
            l[2*r] = 4*gap['svc']
            l[2*t] = 4*gap['svc']
            l[2*w] = 4*gap['svc']
            l[2*v] = 4*gap['svc']
            X.append(l)
            
        for (i,j,k,r,t,f,w,v) in combinations(range(n_potential_bus),8):
            l = [0]*2*n_potential_bus
            l[2*i] = 0.5*gap['svc']
            l[2*j] = 0.5*gap['svc']
            l[2*k] = 0.5*gap['svc']
            l[2*r] = 0.5*gap['svc']
            l[2*t] = 0.5*gap['svc']
            l[2*f] = 0.5*gap['svc']
            l[2*w] = 0.5*gap['svc']
            l[2*v] = 0.5*gap['svc']
            X.append(l)
            l = [0]*2*n_potential_bus
            l[2*i+1] = 0.5*gap['sta']
            l[2*j+1] = 0.5*gap['sta']
            l[2*k+1] = 0.5*gap['sta']
            l[2*r+1] = 0.5*gap['sta']
            l[2*t+1] = 0.5*gap['sta']
            l[2*f+1] = 0.5*gap['sta']
            l[2*w+1] = 0.5*gap['sta']
            l[2*v+1] = 0.5*gap['sta']
            X.append(l) 
            l = [0]*2*n_potential_bus
            l[2*i] = 0.5*gap['svc']
            l[2*j] = 0.5*gap['svc']
            l[2*f] = 0.5*gap['svc']
            l[2*k+1] = 0.5*gap['sta']
            l[2*r+1] = 0.5*gap['sta']
            l[2*t+1] = 0.5*gap['sta']
            l[2*w+1] = 0.5*gap['sta']
            l[2*v+1] = 0.5*gap['sta']
            X.append(l)
            l = [0]*2*n_potential_bus
            l[2*i+1] = 0.5*gap['sta']
            l[2*j+1] = 0.5*gap['sta']
            l[2*f+1] = 0.5*gap['sta']
            l[2*k] = 0.5*gap['svc']
            l[2*r] = 0.5*gap['svc']
            l[2*t] = 0.5*gap['svc']
            l[2*w] = 0.5*gap['svc']
            l[2*v] = 0.5*gap['svc']
            X.append(l) 
            '''
#以上部分为无规则下的样本，一下部分按照一定规则生成样本
        for i1 in range(2):
            for i2 in range(2):
                for i3 in range(2):
                    for i4 in range(2):
                        for i5 in range(2):
                            for i6 in range(2):
                                for i7 in range(2):
                                    for i8 in range(2):
                                        for k in [0.5,1,1.5,2,2.5,3,3.5,4]:
                                            l = [0]*2*n_potential_bus
                                            if i1==0:
                                                l[2*0+i1] = k*gap['svc']
                                            elif i1==1:
                                                l[2*0+i1] = k*gap['sta']
                                            if i2==0:
                                                l[2*1+i2] = k*gap['svc']
                                            elif i2==1:
                                                l[2*1+i2] = k*gap['sta']
                                            if i3==0:
                                                l[2*2+i3] = k*gap['svc']
                                            elif i3==1:
                                                l[2*2+i3] = k*gap['sta']
                                            if i4==0:
                                                l[2*3+i4] = k*gap['svc']
                                            elif i4==1:
                                                l[2*3+i4] = k*gap['sta']
                                            if i5==0:
                                                l[2*4+i5] = k*gap['svc']
                                            elif i5==1:
                                                l[2*4+i5] = k*gap['sta']
                                            if i6==0:
                                                l[2*5+i6] = k*gap['svc']
                                            elif i6==1:
                                                l[2*5+i6] = k*gap['sta']                                                
                                            if i7==0:
                                                l[2*6+i7] = k*gap['svc']
                                            elif i7==1:
                                                l[2*6+i7] = k*gap['sta']     
                                            if i8==0:
                                                l[2*7+i8] = k*gap['svc']
                                            elif i8==1:
                                                l[2*7+i8] = k*gap['sta']
                                            X.append(l) 
                                                     
                                        
                                        
      
   
        return X
    @staticmethod
    def one_sample(bus_potential,one_sample):
        """
        ||将sample_point产生的样本数据中任一样本转化为FormCard.dy_var函数所需参数形式
        """
        types = list()
        rates = list()
        install_index = list()
        for i in range(len(bus_potential)):
            if one_sample[i*2] != 0:
                types.append('svc')
                rates.append(str(one_sample[i*2]))
                install_index.append(bus_potential.index[i])
            if one_sample[i*2+1] != 0:
                types.append('statcom')
                rates.append(str(one_sample[i*2+1]))
                install_index.append(bus_potential.index[i]) 
        return types, rates, install_index
    @staticmethod    
    def train_data(path_train_data, bus_potential, smp_key, path_swi, dynamicload_prob, df_swi_dyld,path_flt,path_bse,path_result,path_batbpa,load_prob, gap={'svc':100,'sta':100}, max_rate={'svc':400,'sta':400}):
        """
        ||对各样本进行计算，生成训练样本数据
        """
        sp = Regis.sample_point(bus_potential, gap, max_rate)
        f = open(path_train_data,'w')
        for i in range(len(sp)):
            print i,len(sp)
            types, rates, install_index = Regis.one_sample(bus_potential,sp[i])
            df_dyvar_card = FormCard.dy_var(types, rates, list(bus_potential['bus_name'][install_index].values), list(bus_potential['vbase'][install_index].values))
            FormFile.swi_file_dynamic_var(path_swi, dynamicload_prob, df_swi_dyld, df_dyvar_card)
            Call.multi_batbpa(path_swi,path_flt,path_bse,path_result,path_batbpa,load_prob, dynamicload_prob)  #批处理稳定计算
            ind = CalculateIndex.technology(smp_key)  
            ind_prob = ind['ind_prob']
            for j in sp[i]:
                f.write(str(j))
                f.write(',')
            f.write(str(ind_prob))
            f.write('\n')
        f.close()
    @staticmethod    
    def learn(train_sample,X=[],y=[]):
        """
        ||从样本数据中学习
        """
        if X==[] and y==[]:
            X1 = np.array(train_sample.iloc[:,0:-1])
            y1 = np.array(train_sample.iloc[:,-1])*1e8
        else:
            X1 = X
            y1 = y
        
        min_max_scaler = preprocessing.MinMaxScaler()
        X1_minmax =  min_max_scaler.fit_transform(X1)
        svr_rbf = SVR(kernel='rbf',C=1e2, gamma = 0.2)
        y_rbf = svr_rbf.fit(X1_minmax, y1)
        func_surrogate = y_rbf
        
        return X1,y1,func_surrogate
        

class Optim:
    """
    优化过程类
    """
    @staticmethod   
    def data_transform(algo_result, delta):
        """
        ||将优化程序程序数据转换为需要的格式，NSGAII
        """
        X = list()  #优化变量
        y1 = list() #费用
        y2 = list() #技术指标
        for i in range(len(algo_result)):
            v = list()
            for j in range(len(algo_result[i].variables)/2):
                v.append(algo_result[i].variables[2*j][0]*sum(algo_result[i].variables[2*j+1])*delta)
                v.append(algo_result[i].variables[2*j][1]*sum(algo_result[i].variables[2*j+1])*delta)
            X.append(v)
            y1.append(algo_result[i].objectives[0])
            y2.append(algo_result[i].objectives[1])
        nondomi_index = list()
        for i in range(len(y1)):
            flag = 0
            for j in range(0,len(y1)):
                if (y1[j]<y1[i] and y2[j]<y2[i]) or (y1[j]<y1[i] and y2[j]==y2[i]) or (y1[j]==y1[i] and y2[j]<y2[i]):
                    flag = 1
                    break
            if flag == 0:
                nondomi_index.append(i)
        X = np.array(X)
        X = X[nondomi_index]
        y1 = np.array(y1)
        y1 = y1[nondomi_index]
        y2 = np.array(y2)
        y2 = y2[nondomi_index]
        return X, y1, y2
    @staticmethod
    def find_add_point_1(X_optimal, y2, train_sample, bus_potential, smp_key, path_swi, dynamicload_prob, df_swi_dyld,\
    path_flt,path_bse,path_result,path_batbpa,load_prob,segment_number=5):
        """
        ||找到第一组新增数据点，将数据点加入train_sample中
        ||基本思想为将当前优化得到的pareto前端在f2维度上等分为segment个区间，每各区间随机选一个点做为新增训练数据点
        """
        #得到每一段的上下界
        gap_lower = list()
        gap_upper = list()
        for i in range(0,int(segment_number)):
            gap_lower.append( min(y2)+i*(max(y2)-min(y2))/segment_number)
            gap_upper.append( min(y2)+(i+1)*(max(y2)-min(y2))/segment_number)
        #将pareto最优解划分入每段
        X_optimal_seg = [[] for i in range(segment_number)]
        y2_seg = [[] for i in range(segment_number)]
        for i in range(len(y2)):
            for j in range(segment_number):
                if y2[i]>=gap_lower[j] and y2[i]<=gap_upper[j]:
                    X_optimal_seg[j].append(X_optimal[i].tolist())
                    y2_seg[j].append(y2[i])
                    break
        X_add = list()
        for i in range(segment_number):
            r = random.randrange(len(X_optimal_seg[i]))
            X_add.append(X_optimal_seg[i][r])
        y2_add_train = list()
        for i in range(len(X_add)):
            types, rates, install_index = Regis.one_sample(bus_potential,X_add[i])
            df_dyvar_card = FormCard.dy_var(types, rates, list(bus_potential['bus_name'][install_index].values), list(bus_potential['vbase'][install_index].values))
            FormFile.swi_file_dynamic_var(path_swi, dynamicload_prob, df_swi_dyld, df_dyvar_card)
            Call.multi_batbpa(path_swi,path_flt,path_bse,path_result,path_batbpa,load_prob, dynamicload_prob)  #批处理稳定计算
            ind = CalculateIndex.technology(smp_key)  
            ind_prob = ind['ind_prob']
            y2_add_train.append(ind_prob)
            X_add[i].append(ind_prob)
        X_add = DataFrame(X_add)
        train_sample = train_sample.append(X_add, ignore_index=True)
   
        return train_sample
    
       

        
