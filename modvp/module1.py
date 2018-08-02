# -*- coding: UTF-8 -*-
"""
Created on Tue Dec 15 21:24:53 2015
@author: T.Han

#包含动态无功补偿优化配置程序中的全部子块

"""
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from os import system, getcwd, path
import win32api
import win32con
import win32gui
import subprocess
from time import sleep
import module2  #自定义模拟键盘窗口输入模块
from pandas import DataFrame, read_csv
from scipy import integrate,pi,exp, optimize, stats
import matplotlib.pyplot as plt
from math import ceil
import numpy as np
from sklearn.cluster import KMeans
#______________________________________________________________________________
#______________________________________________________________________________
class Sample:
    """
    ||生成故障样本类
    """
    @staticmethod
    def __cdf_cleartime__(x, mu_relay, sigma_relay, a_breaker, b_breaker):
        """
        ||故障切除时间的累积概率密度函数
        ||x - 故障切除时间
        ||mu_relay - 继电保护延时服从正态分布的期望值,单位为周波
        ||sigma_relay - 继电保护延时服从正态分布的标准差
        ||a_breaker - 断路器动作时间服从均匀离散分布的小值
        ||b_breaker - 断路器动作时间服从均匀离散分布的大值
        """
        sigma = 2*sigma_relay
        a = a_breaker + mu_relay
        b = b_breaker + mu_relay

        return integrate.quad(lambda x : 1/(2*sigma*(2*pi)**0.5)*exp(-((x-a)**2)/(2*sigma**2))+ \
        1/(2*sigma*(2*pi)**0.5)*exp(-((x-b)**2)/(2*sigma**2)),0,x)[0]
    @staticmethod        
    def __ppf_cleartime__(p, mu_relay, sigma_relay, a_breaker, b_breaker):
        """
        ||故障切除时间概率分布的累积概率密度函数的逆函数
        ||p - 累积概率值
        ||mu_relay - 继电保护延时服从正态分布的期望值,单位为周波
        ||sigma_relay - 继电保护延时服从正态分布的标准差
        ||a_breaker - 断路器动作时间服从均匀离散分布的小值
        ||b_breaker - 断路器动作时间服从均匀离散分布的大值
        """
        return optimize.fsolve(lambda x: Sample.__cdf_cleartime__(x, mu_relay, sigma_relay, a_breaker, b_breaker) - p, [mu_relay+(a_breaker+b_breaker)/2])[0]
    @staticmethod
    def cal_loadlevel(n = 5, mu = 1, sigma = 0.05, p_critical = 0.001):
        """
        ||生成离散化的负荷水平概率分布，返回df格式数据
        ||n - 离散化点数
        ||mu - 负荷水平服从正态分布的期望值，单位为dat文件中的总负荷量
        ||sigma - 负荷水平服从正态分布的标准差
        ||p_critical - 等间距离散化时的边界点累积概率，左侧临界点为p_critical，右侧临界点为1-p_critical
        """
        critical_1 = stats.norm.ppf(p_critical, mu, sigma)
        critical_2 = stats.norm.ppf(1 - p_critical, mu, sigma)
        
        l_cut = list()  #等间距离散化分割点负荷水平
        l = list() #离散化样本点负荷水平
        p_l = list()  #离散样本点的概率值
        for i in range(0,n+1):
            l_cut.append(critical_1 + i * (critical_2 - critical_1)/n)
        for i in range(0,n):
            l.append((l_cut[i] + l_cut[i+1])/2)
        l_cut[0] = float('-inf')
        l_cut[-1] = float('inf')
        for i in range(0,n):
            p_l.append(stats.norm.cdf(l_cut[i+1], mu, sigma) - stats.norm.cdf(l_cut[i], mu, sigma))
            
        loadlevel_prob = DataFrame({'load_level':l, 'prob':p_l}) 
        return loadlevel_prob
    @staticmethod     
    def cal_dynamicload(n = 5, mu = 0.5, sigma = 0.05, p_critical = 0.001):
        """
        ||生成离散化动态负荷比例概率分布，返回df格式数据
        ||n - 离散化点数
        ||mu - 动态负荷比例服从正态分布的期望值，单位为1
        ||sigma - 动态负荷比例服从正态分布的标准差
        ||p_critical - 等间距离散化时的边界点累积概率，左侧临界点为p_critical，右侧临界点为1-p_critical
        """

        critical_1 = stats.norm.ppf(p_critical, mu, sigma)
        critical_2 = stats.norm.ppf(1 - p_critical, mu, sigma)
        
        d_cut = list()  #等间距离散化分割点负荷水平
        d = list() #离散化样本点负荷水平
        p_d = list()  #离散样本点的概率值
        for i in range(0,n+1):
            d_cut.append(critical_1 + i * (critical_2 - critical_1)/n)
        for i in range(0,n):
            d.append((d_cut[i] + d_cut[i+1])/2)
        d_cut[0] = float('-inf')
        d_cut[-1] = float('inf')
        for i in range(0,n):
            p_d.append(stats.norm.cdf(d_cut[i+1], mu, sigma) - stats.norm.cdf(d_cut[i], mu, sigma))
            
        dynamicload_prob = DataFrame({'dynamic_load':d, 'prob':p_d}) 
        return dynamicload_prob
    @staticmethod
    def cal_branch(path_dat,path_fault_rate,dur=1,t_base=3600):
        """
        ||生成支路故障概率分布信息，包括首末节点名称/基准电压/回路标志/故障概率，返回df格式数据
        ||path_dat - 潮流文件路径
        ||path_fault_rate - 支路故障率文件路径，与潮流文件中支路出现的顺序一致
        ||dur - 考虑的时间尺度，单位/hour
        ||t_base - 故障率的时间单位，单位/hour
        """
        df_dat = read_csv(path_dat, encoding='GBK', sep='\n', header=None, dtype=str, na_filter=False) 
        fault_rate = read_csv(path_fault_rate, encoding='GBK', sep='\n', header=None, dtype=float, na_filter=False)
        branch_num = [i for i in range(1,len(fault_rate)+1)]
        from_bus = list()  #支路首端节点名称
        from_vbase = list()  #支路首端电压基准值
        to_bus = list()  # 支路末端节点名称
        to_vbase = list()  # 支路末端电压基准值
        flag_p = list()  #支路并联标志
        for i in df_dat.index:
            if (df_dat[0][i]!='' and df_dat[0][i][0] == 'L') or (df_dat[0][i]!='' and df_dat[0][i][0] == 'T'):
                from_bus.append(df_dat[0][i][6:14].strip())
                from_vbase.append(df_dat[0][i][14:18])
                to_bus.append(df_dat[0][i][18:27].strip())
                to_vbase.append(df_dat[0][i][27:31])
                flag_p.append(df_dat[0][i][31])
        fault_prob = 1-exp((-fault_rate/t_base*dur*dur))     
        branch_prob  = DataFrame({'branch_num':branch_num, 'from_bus':from_bus, 'from_vbase':from_vbase, 'to_bus':to_bus, \
        'to_vbase':to_vbase, 'flag_p':flag_p, 'prob':list(fault_prob[0])})  #dataframe形式的支路信息
        return branch_prob
#______________________________________________________________________________
    @staticmethod
    def cal_cleartime(n=5, mu_relay = 3.4, sigma_relay = 0.1, a_breaker = 1.5, b_breaker = 2, p_critical = 0.001):
        """
        ||生成离散化的故障切除时间概率分布，返回df格式数据
        ||n - 离散化点数
        ||mu_relay - 继电保护延时服从正态分布的期望值,单位为周波
        ||sigma_relay - 继电保护延时服从正态分布的标准差
        ||a_breaker - 断路器动作时间服从均匀离散分布的小值
        ||b_breaker - 断路器动作时间服从均匀离散分布的大值
        ||p_critical - 等间距离散化时的边界点累积概率，左侧临界点为p_critical，右侧临界点为1-p_critical
        """
        critical_1 = Sample.__ppf_cleartime__(p_critical, mu_relay, sigma_relay, a_breaker, b_breaker)
        critical_2 = Sample.__ppf_cleartime__(1 - p_critical, mu_relay, sigma_relay, a_breaker, b_breaker)
        critical_3 = Sample.__ppf_cleartime__(1, mu_relay, sigma_relay, a_breaker, b_breaker)
        
        t_cut = list()  #等间距离散化分割点负荷水平
        t = list() #离散化样本点负荷水平
        p_t = list()  #离散样本点的概率值
        for i in range(0,n+1):
            t_cut.append(critical_1 + i * (critical_2 - critical_1)/n)
        for i in range(0,n):
            t.append(round((t_cut[i] + t_cut[i+1])/2,1))
        t_cut[0] = 0
        t_cut[-1] = critical_3
        for i in range(0,n):
            p_t.append(Sample.__cdf_cleartime__(t_cut[i+1], mu_relay, sigma_relay, a_breaker, b_breaker) \
            - Sample.__cdf_cleartime__(t_cut[i], mu_relay, sigma_relay, a_breaker, b_breaker))

        cleartime_prob = DataFrame({'cleartime':t, 'prob':p_t}) 
        return cleartime_prob
    @staticmethod
    def part_space(branch_prob, type_prob, location_prob, cleartime_prob, reclosure_prob = 0.17):
        """
        ||生成不考虑负荷水平和动态负荷比例不确定性的样本空间，返回df格式数据
        ||branch_prob, type_prob, location_prob, cleartime_prob - 均为df格式数据
        ||reclosure_prob - 自动重合闸不成功率
        """
        branch_num = list()
        fault_type = list()
        location = list()
        clear_time = list()
        prob = list()
        
        for i_branch in branch_prob.index:
            for i_type in type_prob[type_prob.vlevel==float(branch_prob.to_vbase[i_branch])].index:
                for i_location in location_prob[location_prob.vlevel==float(branch_prob.to_vbase[i_branch])].index:
                    for i_cleartime in cleartime_prob.index:
                        branch_num.append(branch_prob.branch_num[i_branch])
                        fault_type.append(type_prob.fault_type[i_type])
                        location.append(location_prob.location[i_location])
                        clear_time.append(cleartime_prob.cleartime[i_cleartime])
                        prob.append(branch_prob.prob[i_branch] * type_prob.prob[i_type] * location_prob.prob[i_location] \
                                * cleartime_prob.prob[i_cleartime] * reclosure_prob)

        smp = DataFrame({'branch_num':branch_num, 'fault_type':fault_type, \
        'location':location, 'clear_time':clear_time, 'prob':prob})
        print('part样本生成成功')
        return smp

#______________________________________________________________________________
    @staticmethod
    def whole_space(load_prob, dynamicload_prob, branch_prob, type_prob, location_prob, cleartime_prob, reclosure_prob = 0.17):
        """
        ||生成整体样本空间，返回df格式数据
        ||load_prob, dynamicload_prob, branch_prob, type_prob, location_prob, cleartime_prob - 均为df格式数据
        ||reclosure_prob - 自动重合闸不成功率
        """
        load_level = list()
        dynamic_load = list()
        branch_num = list()
        fault_type = list()
        location = list()
        clear_time = list()
        prob = list()
        
        for i_load in load_prob.index:
            for i_dynamic in dynamicload_prob.index:
                for i_branch in branch_prob.index:
                    for i_type in type_prob[type_prob.vlevel==float(branch_prob.to_vbase[i_branch])].index:
                        for i_location in location_prob[location_prob.vlevel==float(branch_prob.to_vbase[i_branch])].index:
                            for i_cleartime in cleartime_prob.index:
                                load_level.append(load_prob.load_level[i_load])
                                dynamic_load.append(dynamicload_prob.dynamic_load[i_dynamic])
                                branch_num.append(branch_prob.branch_num[i_branch])
                                fault_type.append(type_prob.fault_type[i_type])
                                location.append(location_prob.location[i_location])
                                clear_time.append(cleartime_prob.cleartime[i_cleartime])
                                prob.append(load_prob.prob[i_load] * dynamicload_prob.prob[i_dynamic]\
                                * branch_prob.prob[i_branch] \
                                * type_prob.prob[i_type] * location_prob.prob[i_location] \
                                * cleartime_prob.prob[i_cleartime] * reclosure_prob)

        smp = DataFrame({'load_level':load_level, 'dynamic_load':dynamic_load, 'branch_num':branch_num, 'fault_type':fault_type, \
        'location':location, 'clear_time':clear_time, 'prob':prob})
        print('whole样本生成成功')
        return smp
#______________________________________________________________________________                        
#______________________________________________________________________________

class FormCard:
    """
    ||生成BPA程序卡的类
    """
    @staticmethod
    def __card_data__(card_name):
        """
        ||以字典形式存储卡位数据，返回卡名称对应的位数据
        ||card_name - 卡名称，如FLT（TYP=3）的卡名称为'FLT_TYP_3'
        """
        column = dict()
        column['FLT_TYP_3']=[[1,5,13,18,26,30,32,35,37,39,43,47,51,53,58,63,67,71,75],\
        [3,12,16,25,29,30,33,35,37,42,46,50,52,57,62,66,70,74,78]]  #单永故障卡位数据
        column['LS_MDE_9'] = [[1,5,13,19,27,32,36,40,46,52,58,65,68,70,71,76],\
        [2,12,16,26,30,32,37,45,51,57,63,66,68,70,75,80]]      
        column['PA'] = [[1,2,10,16,22,28],[1,2,14,20,26,32]]  #潮流计算发电机出力负荷百分比修改卡位数据
        column['V'] = [[1,4,12,16,17,20,23,27,31,32,33,36,39,43,47,51,55,59,63,66,69,77],\
        [1,11,15,16,19,22,26,30,31,32,35,38,42,46,50,54,58,62,65,68,76,80]]  #SVC卡位数据
        column['VG'] = [[1,4,12,17,22,26,30,34,38,42,46,50,54,58,63,69,77],\
        [2,11,15,21,25,29,33,37,41,45,49,53,57,62,67,76,80]]  #STATCOM卡1位数据
        column['VG+'] = [[1,4,12,17,22,27,32,42],[3,11,15,21,26,31,36,45]]  #STATCOM卡2位数据
        return column[card_name]
    @staticmethod
    def __card_str__(card_name, card_value):
        """
        ||形成卡数据并以字符串形式返回
        ||card_name - 卡名称
        ||card_value - 卡值数据，顺序与BPA手册中卡值数据顺序一致，缺省的数据应至少赋一个空格，type=list
        """
        column = FormCard.__card_data__(card_name)
        s = ' '*100  #定义卡字符串
        s = list(s)  #为进行按位替换赋值，将字符串转换为列表
        for i in range(0,len(card_value)):
            s[column[0][i]-1:column[1][i]] = (card_value[i]+' '*10)[0:column[1][i]-column[0][i]+1]
        s = ''.join(s)
        s = s[0:column[1][-1]]
        return s
    @staticmethod  
    def sc_fault(fault_type, bus_a, b1, bus_b, b2, par, cycle, perct, cycle_0 = 0):
        """
        ||形成短路故障卡，不考虑重合闸过程，假设故障一定时间后直接三相断开
        ||fault_type - 短路故障类型，1/2/3/4分别表示单相/两相/三相/两相接地短路
        ||cycle - 故障切除时间，从故障发生后计起，单位为周波
        ||par - 回路编号
        ||perct - 故障位置距前侧节点百分比
        ||cycle_0 - 故障发生时刻，单位为周波，默认为‘0’
        """
        fault_type, bus_a, b1, bus_b, b2, par, cycle, perct, cycle_0 = \
        str(int(fault_type)), str(bus_a), str(b1), str(bus_b), str(b2), str(par), str(cycle), str(perct), str(cycle_0)
        card_name = 'LS_MDE_9'
        card_type = 'LS'
        fault_r, fault_x ,fault_rd , fault_xd = '0','0','0','0'  #不计故障阻抗
        sid1 = '1' #短路故障发生侧
        sid2 = '3' #用断线模拟断路器断开时的故障侧，两侧同时断开
        perct_0 = ' ' #三相断线时的故障点百分比
        mde1, mde2 = '9','-9'  #复故障发生消失标志
        nde = '6' #三相断线形式标志
        pha = '1' #故障相，单相短路设为A相，两相短路或两相接地短路设为BC相，亦表示三相对称故障
        cycle_1 = str(float(cycle_0)+float(cycle))  #从仿真初始时刻计起的故障切除时间
        card_value_1 = [card_type, bus_a, b1, bus_b, b2, par, mde1, cycle_0, fault_r, fault_x, perct, fault_type, pha, sid1, \
        fault_rd, fault_xd]
        card_value_2 = [card_type, bus_a, b1, bus_b, b2, par, mde2, cycle_1, fault_r, fault_x, perct, fault_type, pha, sid1, \
        fault_rd, fault_xd]
        card_value_3 = [card_type, bus_a, b1, bus_b, b2, par, mde1, cycle_1, fault_r, fault_x, perct_0, nde, pha, sid2, \
        fault_rd, fault_xd]
        
        card_1 = FormCard.__card_str__(card_name, card_value_1)
        card_2 = FormCard.__card_str__(card_name, card_value_2)
        card_3 = FormCard.__card_str__(card_name, card_value_3)
        
        card = card_1 + '\n' + card_2 + '\n' + card_3
        return card
    @staticmethod  
    def pa(factor_pl, factor_ql, factor_pg, factor_qg):
        """
        ||形成全区发电机出力负荷百分比修改卡
        ||factor_pl, factor_ql, factor_pg, factor_qg - 四个修改因子
        """
        factor_pl, factor_ql, factor_pg, factor_qg = str(factor_pl), str(factor_ql), str(factor_pg), str(factor_qg)
        card_name = 'PA'
        card_type1 = 'P'
        card_type2 = 'A'
        card_value = [card_type1, card_type2, factor_pl, factor_ql, factor_pg, factor_qg]
        card = FormCard.__card_str__(card_name, card_value)
        return card
    @staticmethod       
    def dy_var(types, rates, bus_names, vbases,\
    parameter_svc = ['V','BUS-21','345',' ',' ','.05',' ','.001','1','1',' ',' ','500',' ','1','1','-1','-1','.1',' ',' ',' '],\
    parameter_statcom = [['VG','BUS-26','345','200','.001','.001','.001','.001','.001','.001','.001','500','500','.001',' ',' ',' ' ],\
    ['VG+','BUS-26','345','1.05','0','1.0','1.0',' ']], mvabase = '100'):
        """
        ||生成动态无功补偿装置卡，以dataframe格式返回，卡数大于等于1
        ||types, rates, bus_names, vbases - 补偿装置的信息，均为列表形式
        ||parameter_svc - svc卡参数
        ||parameter_statcom - statcom卡参数
        """
        list_dyvar_card = list()
        for i in range(0,len(types)):
            if types[i] == 'svc':
                parameter_svc[1] = bus_names[i]
                parameter_svc[2] = vbases[i]
                parameter_svc[14] = str(float(rates[i])/float(mvabase))
                parameter_svc[15] = str(float(rates[i])/float(mvabase))
                parameter_svc[16] = str(-float(rates[i])/float(mvabase))
                parameter_svc[17] = str(-float(rates[i])/float(mvabase))
                list_dyvar_card.append(FormCard.__card_str__('V', parameter_svc))
            elif types[i] == 'statcom':
                parameter_statcom[0][1] = bus_names[i]
                parameter_statcom[0][2] = vbases[i]
                parameter_statcom[0][3] = rates[i]
                list_dyvar_card.append(FormCard.__card_str__('VG', parameter_statcom[0]))
                parameter_statcom[1][1] = bus_names[i]
                parameter_statcom[1][2] = vbases[i]
                list_dyvar_card.append(FormCard.__card_str__('VG+', parameter_statcom[1]))
        df_dyvar_card = DataFrame(list_dyvar_card)
        return df_dyvar_card
  
  
class FormFile:
    @staticmethod
    def __find_str__(df, s):
        """
        ||查找出df中含有字符串s的行的index值，要求只能有一处出现s，若有多处s，返回第一次出现处的index值
        ||df - 读入文件形成的Datafram数据，df[0][index]形式
        """
        for i in df.index:
            if df[0][i].find(s) != -1:
                break
        return i
    @staticmethod              
    def bse_file(path_pfnt, path_dat_changes, load_prob):
        """
        ||生成不同负荷水平对应的潮流计算结果.BSE文件，命名为原始为原bse文件名+’-index‘
        ||path_pfnt - 潮流计算程序路径
        ||path_dat_changes - 生成bse文件所用dat文件
        ||load_prob - 负荷水平概率信息，df格式
        """
        df_dat = read_csv(path_dat_changes, encoding='GBK', sep='\n', header=None, dtype=str, na_filter=False) 
        index_new_bse = FormFile.__find_str__(df_dat, 'NEW_BASE')  #生成.BSE文件名所在处index值
        index_change_load = FormFile.__find_str__(df_dat, 'PA')  #修改负荷卡所在处index值
        s1 = df_dat[0][index_new_bse]  #记录原始值
        s2 = df_dat[0][index_change_load]   #记录原始值
        s3 = list(df_dat[0][index_new_bse].partition('.'))
        order_str =  path_pfnt+' '+path_dat_changes
        
        for i in load_prob.index:
            s3[1] = '-'+str(i)+'.' #得到修改后值，在原始bse文件名后加‘-index’，index与load_prob中index值相同
            df_dat[0][index_new_bse] = ''.join(s3)   #赋修改后值
            df_dat[0][index_change_load] = FormCard.pa(load_prob.load_level[i],load_prob.load_level[i],load_prob.load_level[i],load_prob.load_level[i] )
            df_dat.to_csv(path_dat_changes, sep='\n',encoding='GBK', index=False, header=False)
            a = subprocess.Popen(order_str)
            sleep(2)  #潮流计算程序计算完成后不会自动关闭，采取延时强制关闭
            a.kill
            print('负荷水平'+str(load_prob.load_level[i])+'下.BSE文件生成成功')
            """
            while True:
                sleep(0.01)
                Flag = system('tasklist|find /i "pfnt.exe"')
                if Flag:
                    break
            """
            df_dat[0][index_new_bse] = s1
            df_dat[0][index_change_load] = s2
        print('所有负荷水平下的.BSE文件生成成功')
        df_dat.to_csv(path_dat_changes, sep='\n',encoding='GBK', index=False, header=False)     
        return None
    @staticmethod       
    def swi_file_dynamic_load(path_swi, dynamicload_prob):
        """
        ||生成不同动态负荷比例对应的.SWI文件,将所有.SWI文件的dataframe数据存储于列表中返回
        ||path_swi - 原始swi文件所在路径
        ||动态负荷比例概率信息，df格式
        """
        df_swi = read_csv(path_swi, encoding='GBK', sep='\n', header=None, dtype=str, na_filter=False)
        index_mj = FormFile.__find_str__(df_swi, 'MJ') 
        mj_str = df_swi[0][index_mj]
        s = list(path_swi.partition('.'))
        df_swi_dyld = list()  #存储不同动态负荷对应swi文件的dataframe格式数据
        for i in dynamicload_prob.index:
            s[1] = '-'+str(i)+'.'
            path_new_swi = ''.join(s)   #不同动态负荷比例下swi文件输出路径
            df_swi[0][index_mj] = mj_str[:22]+ str(dynamicload_prob.dynamic_load[i])[0:3]+ mj_str[25:85]
            df_swi.to_csv(path_new_swi, sep='\n',encoding='GBK', index=False, header=False)
            print('生成动态负荷比例为'+str(str(dynamicload_prob.dynamic_load[i])[1:4])+'的.SWI文件生成成功')
            df_swi_dyld.append(read_csv(path_new_swi, encoding='GBK', sep='\n', header=None, dtype=str, na_filter=False))
        print('所有动态负荷比例对应的.SWI文件生成成功')
        return df_swi_dyld
    @staticmethod      
    def swi_file_dynamic_var(path_swi, dynamicload_prob, df_swi_dyld, df_dyvar_card):
        """
        ||在不同动态负荷下的swi文件中插入动态无功补偿卡
        ||path_swi - 原始swi文件所在路劲
        ||df_swi_dyld - 不同动态负荷对应swi文件的dataframe格式数据形成的列表
        ||df_dyvar_card - dataframe格式的动态无功补偿装置BPA卡数据
        """
        index_vg = FormFile.__find_str__(df_swi_dyld[0], 'VG') 
        s = list(path_swi.partition('.'))
        j = 0  #记录df_swi_dyld的index值
        for i in dynamicload_prob.index:
            s[1] = '-'+str(i)+'.'
            path_new_swi = ''.join(s)   #不同动态负荷比例下swi文件输出路径
            df_swi = df_swi_dyld[j][0:index_vg]
            df_swi = df_swi.append(df_dyvar_card)
            df_swi = df_swi.append(df_swi_dyld[j][index_vg+1:len(df_swi_dyld[j])])
            df_swi.to_csv(path_new_swi, sep='\n',encoding='GBK', index=False, header=False)
            j = j+1
            print('向动态负荷比例为'+str(str(dynamicload_prob.dynamic_load[i])[1:4])+'的.SWI文件中添加动态无功补偿卡成功')
        print('向所有.SWI文件中添加动态无功补偿卡成功')
        
    @staticmethod 
    def __single_flt_file__(path_flt, smp, branch_prob):
        """
        ||生成单个稳定批处理计算的.FLT文件
        ||path_flt - 文件存放路径
        ||smp - 样本,不考虑负荷水平和动态负荷比例的部分样本
        ||branch_prob - 支路信息
        """
        f = open(path_flt,'w')
        for i in smp.index:
            card = FormCard.sc_fault(smp.fault_type[i], branch_prob[branch_prob.branch_num==smp.branch_num[i]].from_bus.values[0], \
            branch_prob[branch_prob.branch_num==smp.branch_num[i]].from_vbase.values[0], \
            branch_prob[branch_prob.branch_num==smp.branch_num[i]].to_bus.values[0], \
            branch_prob[branch_prob.branch_num==smp.branch_num[i]].to_vbase.values[0], \
            branch_prob[branch_prob.branch_num==smp.branch_num[i]].flag_p.values[0], \
            smp.clear_time[i], smp.location[i])
            f.write('STUDY  LTP3S 1 '+str(i)+','+str(i)+'\n')
            f.write(card)
            f.write('\n')
            print('故障场景'+str(i)+'故障卡生成成功')
        f.close()
        print('所有故障场景故障卡生成成功')
    @staticmethod    
    def flt_file(path_flt, load_prob, dynamicload_prob, smp_whole, branch_prob):
        """
        ||生成所有负荷水平及动态负荷比例对应的.FLT文件，各文件中的故障信息相同，仅STUDY中填写的稳定计算生成文件名不同，与整体
        ||样本空间的index系统
        ||path_flt - 文件存放路径
        ||load_prob - 负荷水平概率信息
        ||dynamicload_prob - 动态负荷比例概率信息
        ||smp_whole - 整体原始样本
        ||branch_prob - 支路信息
        """
        s = list(path_flt.partition('.'))
        for i in load_prob.index:
            for j in dynamicload_prob.index:
                s[1] = '-'+str(i)+'-'+str(j)+'.'
                path_flt = ''.join(s)
                FormFile.__single_flt_file__(path_flt, smp_whole[smp_whole.load_level==load_prob.load_level[i]]\
                [smp_whole.dynamic_load==dynamicload_prob.dynamic_load[j]], branch_prob)
                print(str(i)+'-'+str(j)+'.FLT文件生成成功')
        print('所有.FLT文件生成成功')
                         
class Call:
    """
    ||调用BPA批出力计算类
    """
    @staticmethod 
    def batbpa(path_swi,path_flt,path_bse,path_result,path_batbpa):
        """
        ||调用稳定批处理程序进行稳定计算，调用该程序计算时不可执行其他键盘鼠标操作
        ||path_swi,path_flt,path_bse - 计算文件路径
        ||path_result - 结果文件存储路径
        ||path_batbpa - 批处理计算程序路径
        """
        name_win = u"故障并行批计算管理器(右键缩放)-[邮址:SongDW@epri.sgcc.com.cn,时间: Nov  4 2012,11:21:45]"   #批处理计算窗口名称
        subprocess.Popen(path_batbpa)
        sleep(0.6)
        hwnd = win32gui.FindWindow(None,name_win)  #获取批处理窗口句柄
        win32gui.SetForegroundWindow (hwnd)  #将批处理窗口置前并激活
        
        module2.key_input(path_swi)  #稳定数据文件路径
        win32api.keybd_event(module2.VK_CODE['enter'],0,0,0)   #输入回车 
        win32api.keybd_event(module2.VK_CODE['enter'],0,win32con.KEYEVENTF_KEYUP,0)
        sleep(0.001)
        module2.key_input(path_flt)  #故障文件路径
        win32api.keybd_event(module2.VK_CODE['enter'],0,0,0)   #输入回车
        win32api.keybd_event(module2.VK_CODE['enter'],0,win32con.KEYEVENTF_KEYUP,0)
        sleep(0.001)
        module2.key_input(path_bse)  #潮流结果文件
        
        #定位到结果输出路径输出控件
        for i in range(0,7):
            win32api.keybd_event(module2.VK_CODE['tab'],0,0,0)   #输入回车
            win32api.keybd_event(module2.VK_CODE['tab'],0,win32con.KEYEVENTF_KEYUP,0)
            sleep(0.001)
        module2.key_input(path_result) #输入结果输出路径
    
        # ctrl+1开始计算
        win32api.keybd_event(0x11,0,0,0)  #ctrl键 
        win32api.keybd_event(0x31,0,0,0)  #1键  
        win32api.keybd_event(0x31,0,win32con.KEYEVENTF_KEYUP,0) #释放按键  
        win32api.keybd_event(0x11,0,win32con.KEYEVENTF_KEYUP,0)  
        
        while True:
            sleep(1)
            Flag = system('tasklist|find /i "swnts.exe"')
            sleep(1)
            Flag = Flag * system('tasklist|find /i "swnts.exe"')
            sleep(1)
            Flag = Flag * system('tasklist|find /i "swnts.exe"')
            if Flag:
                break
        print('***********')
        sleep(0.5)   ############################################The initial value is 0.5
        k = system('taskkill /IM batBPA.exe /f')
        if k==0:
            print('<Finished a successful batBPA program>')
        else:
            print('<Finished a fail batBPA program>')
    @staticmethod        
    def multi_batbpa(path_swi,path_flt,path_bse,path_result,path_batbpa,load_prob, dynamicload_prob):
        """
        ||多次调用batbpa进行不同负荷水平、不同动态负荷比例下的稳定计算
        """
        s1 = list(path_swi.partition('.'))
        s2 = list(path_flt.partition('.'))
        s3 = list(path_bse.partition('.'))
        for i in load_prob.index:
            for j in dynamicload_prob.index:
                s1[1] = '-'+str(j)+'.'
                s2[1] = '-'+str(i)+'-'+str(j)+'.'
                s3[1] = '-'+str(i)+'.'
                path_swi = ''.join(s1)
                path_flt = ''.join(s2)
                path_bse = ''.join(s3)
                Call.batbpa(path_swi,path_flt,path_bse,path_result,path_batbpa)
                print('完成负荷水平'+str(i)+'-动态负荷比例'+str(j)+'下所有故障稳定计算')
        print('完成所有故障稳定计算')
                
        
        
        
    
class CalculateVoltageIndex:
    """
    ||计算电压评价指标的类
    """
    @staticmethod 
    def __get_tech_info__(err):
        """
        ||从一个.ERR文件中获取节点名称(基准电压)、电压数据位置信息，便于对其他.ERR文件的重复计算,用于技术指标计算
        ||返回 info = {'bus_name': bus_name, 'start_index': start_index, 'end_index': end_index}
        ||path_err - 任意一个.ERR文件的路径
        """
        key_word1 = '输出数据列表'
        key_word2 = '节点'
        info = dict()
        bus_name = list()
        start_index = list()
        end_index = list()
        flag = 0
        for i in err.index:
            if err[0][i].find(key_word1) !=  -1 and err[0][i].find(key_word2) !=  -1 and flag == 0:
                bn_index1 = i  #第一个出现母线名称信息处的索引，之后开始出现节点电压数据
                flag = 1
                continue
            if err[0][i].find(key_word1) !=  -1 and err[0][i].find(key_word2) !=  -1 and flag:
                bn_index2 = i  #第二个出现母线名称信息处的索引，用以确定电压数据的长度
                break
        for i in range(1,len(err)):
            if err[0][err.index[-i]].find(key_word1) !=  -1 and err[0][err.index[-i]].find(key_word2) !=  -1:
                bn_index9 = err.index[-i]
                break  #最后一个出现母线名称信息处的索引        
        for i in range(0,int(1e5)):
            bus_name.append(err[0][bn_index1 + i * (bn_index2 - bn_index1)].partition('"')[2].partition('"')[0].split(' ')[0])
            start_index.append(bn_index1 + i * (bn_index2 - bn_index1) + 4)
            end_index.append(bn_index1 + (i + 1) * (bn_index2 - bn_index1) - 1)
            if bn_index1 + i * (bn_index2 - bn_index1) == bn_index9:
                break
        info = {'bus_name': bus_name, 'start_index': start_index, 'end_index': end_index}
        return info
    @staticmethod          
    def __f_t__(x, u_0, mu):
        """
        ||对节点某时刻电压进行转化计算
        ||x - 某时刻电压值
        ||u_0 - 节点电压初始值
        ||mu - 电压偏差临界阈值
        """
        m = (abs(x-u_0))/u_0
        if m > mu:
            return m
        else:
            return 0
    @staticmethod        
    def __err_u__(err, info):
        """
        ||将从err数据中提取节点电压数据和时间数据到u中并返回
        ||err- err数据，type=DataFrame
        ||info - __get_tech_info__返回的字典
        """
        u = DataFrame()  #存储电压数据，列标包括Index、Time及info['bus_name']，数据为float型
        u[info['bus_name'][0]] = err[0][info['start_index'][0]:info['end_index'][0]+1].values
        u['Time'] =  u[info['bus_name'][0]].apply(lambda x: float(x.partition('	    ')[0]))  #根据err中任一节点电压数据获取时间数据
        for i in range(0,len(info['bus_name'])):  
            u[info['bus_name'][i]] = err[0][info['start_index'][i]:info['end_index'][i]+1].values
            u[info['bus_name'][i]] = u[info['bus_name'][i]].apply(lambda x: float(x.partition('	    ')[2]))  #将节点电压分别赋给u
        return u
    @staticmethod 
    def err_u(path_err):
        """
        ||从路径为path_err的.err文件中提取电压数据,供外部可视化程序调用
        """
        err = read_csv(path_err, encoding='GBK',sep='\n', header=None, dtype=str, na_filter=False)  #读入.ERR文件
        info = CalculateVoltageIndex.__get_tech_info__(err)
        return CalculateVoltageIndex.__err_u__(err, info)
    @staticmethod 
    def __u_f__(u, mu, info):
        """
        ||将节点电压数据按照__f_t__函数进行转换计算, 该函数会直接改变传递参数u的值
        ||u - 节点电压数据，type = DataFrame
        ||mu - 电压偏差临界阈值
        ||info - __get_tech_info__返回的字典
        """
        for i in range(0,len(info['bus_name'])): 
            u_0 = u[info['bus_name'][i]][0]  #节点电压初值
            u[info['bus_name'][i]] = u[info['bus_name'][i]].apply(lambda x: CalculateVoltageIndex.__f_t__(x, u_0, mu))  #按照__f_t__进行转换计算
        return None
    @staticmethod     
    def technology_single(err, info, cycle, mu = 0.2, cycle_d = 500, cycle_0 = 0):
        """
        ||计算单故障的技术指标值
        ||path_err - .ERR文件路径
        ||info - __get_tech_info__返回的字典
        ||cycle - 故障切除时间，从故障发生后计起，单位为周波
        ||cycle_0 - 故障发生时刻，单位为周波，与FormCard.sc_fault中的cycle_0取值一致
        ||cycle_d - 暂态时间尺度，从故障切除时刻计起，单位为周波，默认取500,为保证终止时刻在仿真点中，必须取整数
        ||mu - 电压偏差临界阈值，默认20%
        """
        cycle, cycle_0, cycle_d = float(cycle), float(cycle_0), float(cycle_d)
        u = CalculateVoltageIndex.__err_u__(err, info)  #从err中提取节点电压数据和时间数据
        CalculateVoltageIndex.__u_f__(u, mu, info)  #将u中的节点电压数据按照__f_t__函数进行转换计算
        
        ind = (sum(sum(u.loc[u[u.Time == (cycle_0 + cycle)].index[1]:u[u.Time == (cycle_0 + cycle + cycle_d)].index[0], info['bus_name']].values)))\
        /(len(info['bus_name']) * (u[u.Time == (cycle_0 + cycle + cycle_d)].index[0] - u[u.Time == (cycle_0 + cycle)].index[1] + 1))  
        #技术指标值，各节点故障切除后各时刻u中数据求和除以节点数与暂态时间尺度内的仿真点数
        return ind
    @staticmethod     
    def technology(smp, mu = 0.2, cycle_d = 500, cycle_0 = 0 ):
        """
        ||计算计及各故障场景概率的技术指标,返回各种故障场景ind['ind_smp'] = ind_smp及概率指标值ind['ind_prob'] = ind_prob的字典
        ||smp - 故障场景样本
        ||mu - 电压偏差临界阈值，默认20%
        ||cycle_d - 暂态时间尺度，从故障切除时刻计起，单位为周波，默认取500,为保证终止时刻在仿真点中，必须取整数
        ||cycle_0 - 故障发生时刻，单位为周波，与FormCard.sc_fault中的cycle_0取值一致
        """
        ind_smp = list()   #每种故障场景的技术指标值，存储顺序与smp中故障场景顺序一致
        ind = dict()  #存储各故障场景指标值和概率指标值
        info_set = dict()
        cwd = getcwd()
        for i in smp.index:
            try:
                path_err = path.join(cwd,'result\\'+str(i)+'.ERR')
                err = read_csv(path_err, encoding='GBK',sep='\n', header=None, dtype=str, na_filter=False)  #读入.ERR文件
                if len(err) in info_set.keys():
                    ind_smp.append(CalculateVoltageIndex.technology_single(err, info_set[len(err)], smp.clear_time[i], mu, cycle_d, cycle_0))
                else:
                    info_set[len(err)] = CalculateVoltageIndex.__get_tech_info__(err)
                    ind_smp.append(CalculateVoltageIndex.technology_single(err, info_set[len(err)], smp.clear_time[i], mu, cycle_d, cycle_0))
                print('故障场景'+str(i)+'技术指标计算完成')
            except ValueError:
                path_err = path.join(cwd,'result\\'+str(i)+'.SWX')
                err = read_csv(path_err, encoding='GBK',sep='\n', header=None, dtype=str, na_filter=False)  #读入.ERR文件
                if len(err) in info_set.keys():
                    ind_smp.append(CalculateVoltageIndex.technology_single(err, info_set[len(err)], smp.clear_time[i], mu, cycle_d, cycle_0))
                else:
                    info_set[len(err)] = CalculateVoltageIndex.__get_tech_info__(err)
                    ind_smp.append(CalculateVoltageIndex.technology_single(err, info_set[len(err)], smp.clear_time[i], mu, cycle_d, cycle_0))
                print('故障场景'+str(i)+'技术指标计算完成')
        ind_prob = sum(smp.prob * ind_smp)
        ind['ind_smp'] = ind_smp
        ind['ind_prob'] = ind_prob
        print('概率技术指标计算完成')
        return ind

class CalculateAngleIndex:
    """
    ||计算电压评价指标的类
    """
    @staticmethod 
    def __get_tech_info__(err):
        """
        ||从一个.ERR文件中获取节点名称(基准电压)、电压数据位置信息，便于对其他.ERR文件的重复计算,用于技术指标计算
        ||返回 info = {'bus_name': bus_name, 'start_index': start_index, 'end_index': end_index}
        ||path_err - 任意一个.ERR文件的路径
        """
        key_word1 = '输出数据列表'
        key_word2 = '发电机'
        info = dict()
        bus_name = list()
        start_index = list()
        end_index = list()
        flag = 0
        for i in err.index:
            if err[0][i].find(key_word1) !=  -1 and err[0][i].find(key_word2) !=  -1 and flag == 0:
                bn_index1 = i  #第一个出现母线名称信息处的索引，之后开始出现节点电压数据
                flag = 1
                continue
            if err[0][i].find(key_word1) !=  -1 and err[0][i].find(key_word2) !=  -1 and flag:
                bn_index2 = i  #第二个出现母线名称信息处的索引，用以确定电压数据的长度
                break
        for i in range(1,len(err)):
            if err[0][err.index[-i]].find(key_word1) !=  -1 and err[0][err.index[-i]].find(key_word2) !=  -1:
                bn_index9 = err.index[-i]
                break  #最后一个出现母线名称信息处的索引        
        for i in range(0,int(1e5)):
            bus_name.append(err[0][bn_index1 + i * (bn_index2 - bn_index1)].partition('"')[2].partition('"')[0].split(' ')[0])
            start_index.append(bn_index1 + i * (bn_index2 - bn_index1) + 4)
            end_index.append(bn_index1 + (i + 1) * (bn_index2 - bn_index1) - 1)
            if bn_index1 + i * (bn_index2 - bn_index1) == bn_index9:
                break
        info = {'bus_name': bus_name, 'start_index': start_index, 'end_index': end_index}
        return info
    @staticmethod          
    def __f_t__(x, u_0, mu):
        """
        ||对节点某时刻电压进行转化计算
        ||x - 某时刻电压值
        ||u_0 - 节点电压初始值
        ||mu - 电压偏差临界阈值
        """
        m = (abs(x-u_0))/u_0
        if m > mu:
            return m
        else:
            return 0
    @staticmethod        
    def __err_u__(err, info):
        """
        ||将从err数据中提取节点电压数据和时间数据到u中并返回
        ||err- err数据，type=DataFrame
        ||info - __get_tech_info__返回的字典
        """
        u = DataFrame()  #存储电压数据，列标包括Index、Time及info['bus_name']，数据为float型
        u[info['bus_name'][0]] = err[0][info['start_index'][0]:info['end_index'][0]+1].values
        u['Time'] =  u[info['bus_name'][0]].apply(lambda x: float(x.partition('	    ')[0]))  #根据err中任一节点电压数据获取时间数据
        for i in range(0,len(info['bus_name'])):  
            u[info['bus_name'][i]] = err[0][info['start_index'][i]:info['end_index'][i]+1].values
            u[info['bus_name'][i]] = u[info['bus_name'][i]].apply(lambda x: float(x.partition('	    ')[2]))  #将节点电压分别赋给u
        return u
    @staticmethod 
    def err_u(path_err):
        """
        ||从路径为path_err的.err文件中提取电压数据,供外部可视化程序调用
        """
        err = read_csv(path_err, encoding='GBK',sep='\n', header=None, dtype=str, na_filter=False)  #读入.ERR文件
        info = CalculateAngleIndex.__get_tech_info__(err)
        return CalculateAngleIndex.__err_u__(err, info)
    @staticmethod 
    def __u_f__(u, mu, info):
        """
        ||将节点电压数据按照__f_t__函数进行转换计算, 该函数会直接改变传递参数u的值
        ||u - 节点电压数据，type = DataFrame
        ||mu - 电压偏差临界阈值
        ||info - __get_tech_info__返回的字典
        """
        for i in range(0,len(info['bus_name'])): 
            u_0 = u[info['bus_name'][i]][0]  #节点电压初值
            u[info['bus_name'][i]] = u[info['bus_name'][i]].apply(lambda x: CalculateAngleIndex.__f_t__(x, u_0, mu))  #按照__f_t__进行转换计算
        return None
    @staticmethod     
    def technology_single(err, info, cycle, mu = 0.2, cycle_d = 500, cycle_0 = 0):
        """
        ||计算单故障的技术指标值
        ||path_err - .ERR文件路径
        ||info - __get_tech_info__返回的字典
        ||cycle - 故障切除时间，从故障发生后计起，单位为周波
        ||cycle_0 - 故障发生时刻，单位为周波，与FormCard.sc_fault中的cycle_0取值一致
        ||cycle_d - 暂态时间尺度，从故障切除时刻计起，单位为周波，默认取500,为保证终止时刻在仿真点中，必须取整数
        ||mu - 电压偏差临界阈值，默认20%
        """
        cycle, cycle_0, cycle_d = float(cycle), float(cycle_0), float(cycle_d)
        u = CalculateAngleIndex.__err_u__(err, info)  #从err中提取节点电压数据和时间数据
        CalculateAngleIndex.__u_f__(u, mu, info)  #将u中的节点电压数据按照__f_t__函数进行转换计算
        
        ind = (sum(sum(u.loc[u[u.Time == (cycle_0 + cycle)].index[1]:u[u.Time == (cycle_0 + cycle + cycle_d)].index[0], info['bus_name']].values)))\
        /(len(info['bus_name']) * (u[u.Time == (cycle_0 + cycle + cycle_d)].index[0] - u[u.Time == (cycle_0 + cycle)].index[1] + 1))  
        #技术指标值，各节点故障切除后各时刻u中数据求和除以节点数与暂态时间尺度内的仿真点数
        return ind
    @staticmethod     
    def technology(smp, mu = 0.2, cycle_d = 500, cycle_0 = 0 ):
        """
        ||计算计及各故障场景概率的技术指标,返回各种故障场景ind['ind_smp'] = ind_smp及概率指标值ind['ind_prob'] = ind_prob的字典
        ||smp - 故障场景样本
        ||mu - 电压偏差临界阈值，默认20%
        ||cycle_d - 暂态时间尺度，从故障切除时刻计起，单位为周波，默认取500,为保证终止时刻在仿真点中，必须取整数
        ||cycle_0 - 故障发生时刻，单位为周波，与FormCard.sc_fault中的cycle_0取值一致
        """
        ind_smp = list()   #每种故障场景的技术指标值，存储顺序与smp中故障场景顺序一致
        ind = dict()  #存储各故障场景指标值和概率指标值
        info_set = dict()
        cwd = getcwd()
        for i in smp.index:
            try:
                path_err = path.join(cwd,'result\\'+str(i)+'.ERR')
                err = read_csv(path_err, encoding='GBK',sep='\n', header=None, dtype=str, na_filter=False)  #读入.ERR文件
                if len(err) in info_set.keys():
                    ind_smp.append(CalculateAngleIndex.technology_single(err, info_set[len(err)], smp.clear_time[i], mu, cycle_d, cycle_0))
                else:
                    info_set[len(err)] = CalculateAngleIndex.__get_tech_info__(err)
                    ind_smp.append(CalculateAngleIndex.technology_single(err, info_set[len(err)], smp.clear_time[i], mu, cycle_d, cycle_0))
                print('故障场景'+str(i)+'技术指标计算完成')
            except ValueError:
                path_err = path.join(cwd,'result\\'+str(i)+'.SWX')
                err = read_csv(path_err, encoding='GBK',sep='\n', header=None, dtype=str, na_filter=False)  #读入.ERR文件
                if len(err) in info_set.keys():
                    ind_smp.append(CalculateAngleIndex.technology_single(err, info_set[len(err)], smp.clear_time[i], mu, cycle_d, cycle_0))
                else:
                    info_set[len(err)] = CalculateAngleIndex.__get_tech_info__(err)
                    ind_smp.append(CalculateAngleIndex.technology_single(err, info_set[len(err)], smp.clear_time[i], mu, cycle_d, cycle_0))
                print('故障场景'+str(i)+'技术指标计算完成')
        ind_prob = sum(smp.prob * ind_smp)
        ind['ind_smp'] = ind_smp
        ind['ind_prob'] = ind_prob
        print('概率技术指标计算完成')
        return ind



        

        
class KeyFault:
    """
    判断关键扰动的类
    """
    @staticmethod 
    def _voltage_curve_(smp):
        """ 
        ||显示所有仿真故障的电压变化曲线     
        """
        cwd = getcwd()
        k = ceil(pow(len(smp),0.5))
        fig = plt.figure(figsize=(k*3.2,k*2.4), dpi=200)
        for i in smp.index:
            path_err = path.join(cwd,'result\\'+str(i)+'.ERR')
            u = CalculateVoltageIndex.err_u(path_err)
            ax = plt.subplot(k, k, i+1)
            for c in u.columns:
                if c!='Time':
                    plt.plot(u['Time'], u[c])
            ax.text(300, 0.1, i, fontsize=k*2.6, va='bottom')
        fig.suptitle("Transient Voltage Curves of Whole Faults",fontsize=k*3)
        plt.show() 
    @staticmethod 
    def _bar_pre_(smp, ind):
        """
        ||绘制smp_whole的暂态稳定指标条形图,选择关键扰动前，用于提供参考
        """
        fig = plt.figure(figsize=(16,8), dpi=200)
        plt.bar(smp.index,ind['ind_smp'],align="center")
        plt.xlabel('Number of contingencies(smp_whole.index)',fontsize=16)
        plt.ylabel('Short-term voltage stability level',fontsize=16)
        plt.xticks(smp.index)
        plt.xlim(-1,len(smp)+1)
        fig.suptitle('Transient Voltage Stability Index of Whole Faults',fontsize=21)
        plt.show()
    @staticmethod     
    def _bar_key_(smp, pfault_index, index_key, ind):
        """
        ||绘制smp_whole的暂态稳定指标条形图,标记出关键扰动及忽略扰动
        """
        ind_smp  = DataFrame(ind['ind_smp'])
        fig = plt.figure(figsize=(16,8), dpi=200)
        plt.bar(smp.index,ind_smp[0], label="Non-critical contingencies",align="center")
        plt.bar(index_key, ind_smp[0][index_key], color="red", label="Critical contingencies", align="center")
        plt.bar(pfault_index, ind_smp[0][pfault_index], color="green", label="Ignorent contingencies", align="center")
        
        plt.legend()

        plt.xlabel('Number of contingency branches',fontsize=18)
        plt.ylabel('$f_1|_\phi=n$',fontsize=18)
        plt.xticks(smp.index)
        plt.xlim(-1,len(smp)+1)
        fig.suptitle('Transient Voltage Stability Index of Whole Faults(clustered)',fontsize=21)
        plt.show()
    @staticmethod     
    def _cluster_key_(smp, ind, pfault_index):
        """
        ||聚类得到关键扰动，列表形式返回关键扰动在smp_whole中的index值
        """
        ind_smp = DataFrame(ind['ind_smp'])  #所有故障的ind转化为DataFrame格式
        index_smp = list(smp.index)  #所有样本的index，ind_smp.index=smp_whole.index
        for i in pfault_index:
            index_smp.remove(i)
        X = np.array(ind_smp[0][index_smp])
        X.resize((X.size,1))
        estimators =  KMeans(n_clusters=2)
        estimators.fit(X)
        labels = estimators.labels_
        index_key = list()  #关键扰动在smp_whole中的index
        index_unkey = list() #非关键扰动在smp_whole中的index
        for i in range(0,labels.size):
            if labels[i]==1:
                index_key.append(index_smp[i])
            elif labels[i]==0:
                index_unkey.append(index_smp[i])
        if ind_smp[0][index_key[0]]>ind_smp[0][index_unkey[0]]:
            return index_key
        else:
            return index_unkey
        
    @staticmethod      
    def find_first(smp, path_flt, path_swi, path_bse, load_prob, dynamicload_prob, smp_whole, branch_prob, path_result,path_batbpa):
        """
        ||选择smp中的关键扰动，并将smp中的关键扰动部分返回
        """
        FormFile.flt_file(path_flt, load_prob, dynamicload_prob, smp_whole, branch_prob) #生成初始样本对应的批处理稳定计算.FLT文件
###########################################        
        Call.multi_batbpa(path_swi,path_flt,path_bse,path_result,path_batbpa,load_prob, dynamicload_prob)  #批处理稳定计算
        ind = CalculateVoltageIndex.technology(smp_whole)  #计算初始样本中各故障场景技术指标值

        return smp, ind
    @staticmethod 
    def find_next(pfault_index, ind, smp):
        
        pfault_index = list(pfault_index)
        DataFrame(ind['ind_smp'])
            
        index_key = KeyFault._cluster_key_(smp, ind, pfault_index)  #聚类得到关键扰动在smp中的index值
        return index_key, smp.iloc[index_key]
        
class PotentialBus:
    """
    ||选择灵敏母线作为有潜力的节点
    """
    @staticmethod 
    def _bar_pre_(bus_info, sens):
        """
        ||绘制bus_info中所有节点的灵敏度条形图,选择灵敏母线前，用于提供参考
        """
        fig = plt.figure(figsize=(16,8), dpi=200)
        plt.bar(bus_info.index,sens,align="center")
        plt.xlabel('Number of bus',fontsize=16)
        plt.ylabel('$s_i$',fontsize=16)
        plt.xticks(bus_info.index)
        plt.xlim(-1,len(bus_info)+1)
        plt.ylim(min(sens)-(max(sens)-min(sens))*0.2,max(sens)+(max(sens)-min(sens))*0.2)
        fig.suptitle('Sensitivity of All Buses',fontsize=21)
        plt.show()
    @staticmethod     
    def _cluster_potential_(bus_info, sens, pbus_index):
        """
        ||聚类得到灵敏母线，列表形式返回灵敏母线在bus_info中的index值
        """
        sens = DataFrame(sens)  #所有节点的sens转化为DataFrame格式
        index_bus_info = list(bus_info.index)  #所有节点的index，sens.index=bus_info.index
        #for i in pbus_index:               #####################################################
        #    index_bus_info.remove(i)       #####################################################
        X = np.array(sens[0][index_bus_info])
        X.resize((X.size,1))
        estimators =  KMeans(n_clusters=2)
        estimators.fit(X)
        labels = estimators.labels_
        index_potential = list()  #灵敏母线在bus_info中的index
        index_unpotential = list() #非灵敏母线在bus_info中的index
        for i in range(0,labels.size):
            if labels[i]==1:
                index_potential.append(index_bus_info[i])
            elif labels[i]==0:
                index_unpotential.append(index_bus_info[i])
        if sens[0][index_potential[0]]>sens[0][index_unpotential[0]]:
            return index_potential
        else:
            return index_unpotential
    @staticmethod         
    def _bar_potential_(bus_info, pbus_index, index_potential, sens):
        """
        ||绘制bus_info的灵敏度条形图,标记出灵敏母线及忽略母线
        """
        sens  = DataFrame(sens)
        fig = plt.figure(figsize=(16,8), dpi=200)
        plt.bar(bus_info.index+1,sens[0], label="Non-sensitive buses",align="center")
##############################        
        a =list()
        for i in index_potential:
            a.append(i+1)
        plt.bar(a, sens[0][index_potential], color="red", label="Sensitive buses", align="center")
####################color = green   

        #plt.bar(pbus_index+1, sens[0][pbus_index], label="Ignorent buses", align="center")
        
        plt.legend()

        plt.xlabel('Number of buses',fontsize=18)
        plt.ylabel('$s_i$',fontsize=18)
        plt.xticks(bus_info.index+1)
        plt.xlim(-1,len(bus_info)+1)
        plt.ylim(min(sens[0])-(max(sens[0])-min(sens[0]))*0.2,max(sens[0])+(max(sens[0])-min(sens[0]))*0.2)  
        fig.suptitle('Sensitivity of All Buses(clustered)',fontsize=21)
        plt.show()
    @staticmethod 
    def bus_info(path_dat):
        """
        ||从path_dat中得到系统非发电机节点信息，返回DataFrame形式数据
        """
        df_dat = read_csv(path_dat, encoding='GBK', sep='\n', header=None, dtype=str, na_filter=False) 
        bus_name = list()
        vbase = list()
        for i in df_dat.index:
            if df_dat[0][i]!='' and df_dat[0][i][0] == 'B' and df_dat[0][i][38:57]==' '*19:
                bus_name.append(df_dat[0][i][6:14].strip())
                vbase.append(df_dat[0][i][14:18])      
        bus_info  = DataFrame({'bus_name':bus_name, 'vbase':vbase})   
        return bus_info
        
    @staticmethod       
    def find_first(s_initial, s_delta, path_dat, path_swi, path_flt, path_bse, path_result,path_batbpa, smp_key, load_prob, dynamicload_prob, df_swi_dyld):
        """
        ||选择灵敏母线，返回bus_info中的灵敏母线
        """
        sens = list()   #所有节点的灵敏度信息
        types = ['statcom']
        rates = [str(s_delta)]
        bus_info = PotentialBus.bus_info(path_dat)  #获取系统非发电机节点信息
        for i in bus_info.index:
            df_dyvar_card = FormCard.dy_var(types, rates, [bus_info['bus_name'][i]], [bus_info['vbase'][i]])
            FormFile.swi_file_dynamic_var(path_swi, dynamicload_prob, df_swi_dyld, df_dyvar_card)
            Call.multi_batbpa(path_swi,path_flt,path_bse,path_result,path_batbpa,load_prob, dynamicload_prob)  #批处理稳定计算
            ind = CalculateVoltageIndex.technology(smp_key)  
            sens.append(ind['ind_prob'])   #计算初始样本中各故障场景技术指标值
        return bus_info, sens

    @staticmethod       
    def find_next(bus_info, sens, pbus_index):
        index_potential = PotentialBus._cluster_potential_(bus_info, sens, pbus_index)  #聚类得到灵敏母线在bus_info中的index值
        return index_potential, bus_info.iloc[index_potential]
        
        

      
        
   


    
        

    
    

        
        
        
        
        
    


        
        
            
        
        

        
    
 

            