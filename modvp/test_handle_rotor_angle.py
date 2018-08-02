# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 21:11:03 2016

@author: T.Han
"""
from module1 import CalculateAngleIndex

a=CalculateAngleIndex.err_u(os.path.join(cwd,'result\\43.err'), G, branch_prob,43, Tj)

cycle_start = list()
cycle_end = list()

for i in range(1,len(a1)):
    if a1[i]-a1[i-1]>300:
        cycle_start.append(i)
for i in range(1,len(cycle_start)):
    cycle_end.append(cycle_start[i])
cycle_end.append(len(a1))

for i in range(0,len(cycle_start)):
    a1[cycle_start[i]:cycle_end[i]+1] = a1[cycle_start[i]:cycle_end[i]+1] + (i+1)*360
    


    
        