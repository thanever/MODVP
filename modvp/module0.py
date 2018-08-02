# -*- coding: UTF-8 -*-
"""
Created on Tue Dec 15 21:39:12 2015

@author: Administrator
"""
import sys
reload(sys)
sys.setdefaultencoding('utf8')



from module3 import Regis,Optim
from pandas import read_csv
from platypus.algorithms import NSGAII,random
from platypus.core import Problem
from platypus.types import Binary

def optimization_alpha(path_train_data,path_smp_key,path_bus_potential,smp_key,bus_potential,\
q_max, delta, fixedcost_sta, pcost_sta, fixedcost_svc, pcost_svc,seed_num, population_size,\
max_interation):
    if smp_key==[] and bus_potential==[]:
        smp_key = read_csv(path_smp_key,encoding='GBK',header=0,index_col=0)
        bus_potential = read_csv(path_bus_potential,encoding='GBK',dtype = str,header=0,index_col=0)
    train_sample = read_csv(path_train_data, encoding='GBK', sep=',', dtype=float, na_filter=False, header=None)  #读入训练样本

    X,y,func_surrogate = Regis.learn(train_sample)  


    cost_parameter = {'svc':{'install':fixedcost_svc,'purchase':pcost_svc},\
    'statcom':{'install':fixedcost_sta,'purchase':pcost_sta}}
    def dvaropt(vars):
        total_cost = 0
        constrain = list()
        x_predict = list()
        for i in range(0,len(bus_potential)*2,2):
            r=sum(vars[i+1])*delta
            total_cost = total_cost + vars[i][0] * (cost_parameter['svc']['install'] + r * cost_parameter['svc']['purchase'])\
            + vars[i][1] * (cost_parameter['statcom']['install'] + r * cost_parameter['statcom']['purchase'])
        
                 
            if vars[i][0] == 1 and vars[i][1] == 0 and r!=0:
                x_predict.append(r/float(q_max))
                x_predict.append(0)                
            elif vars[i][1] == 1 and vars[i][0] == 0 and r!=0:
                x_predict.append(0)
                x_predict.append(r/float(q_max))
            elif vars[i][0] == 1 and vars[i][1] == 0 and r==0:
                x_predict.append(0)
                x_predict.append(0)
            elif vars[i][1] == 1 and vars[i][0] == 0 and r==0:
                x_predict.append(0)
                x_predict.append(0)
            elif vars[i][1] == 0 and vars[i][0] == 0 :
                x_predict.append(0)
                x_predict.append(0)
            elif vars[i][1] == 1 and vars[i][0] == 1 :
                x_predict.append(0)
                x_predict.append(0)
            constrain.append(vars[i][0]+vars[i][1])
        ind_prob = func_surrogate.predict(x_predict)[0]*1e-8
        return [total_cost, ind_prob], constrain

    problem = Problem(2*len(bus_potential), 2, len(bus_potential))
    varible_type = list()
    for i in range(0,len(bus_potential)):
        varible_type.append(Binary(2))
        varible_type.append(Binary(int(q_max/delta)))
    problem.types[:] = varible_type
    problem.constraints[:] = "<=1"
    problem.function = dvaropt
    algorithm = NSGAII(problem)
    algorithm.population_size = population_size

    random.seed(seed_num)
    algorithm.run(max_interation*population_size)

    X_optimal, y1, y2 = Optim.data_transform(algorithm.result, delta)  #解析优化算法结果，得到pareto最优解
    return X_optimal, y1, y2

