# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 16:48:16 2016

@author: T.Han
"""
import os
from module1 import Sample, FormFile, KeyFault, PotentialBus, FormCard, Call, CalculateIndex
#from module3 import Regis,Optim
from pandas import read_csv
import matplotlib.pyplot as plt
from platypus.algorithms import NSGAII,random
from platypus.core import Problem
from platypus.types import Binary
import sys
from PyQt4 import QtGui
from DVP_ui import Ui_MainWindow
from pandas import DataFrame
from module0 import optimization_alpha 

 
class DVP_App(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self, parent = None):
        self.cwd = os.getcwd()
        QtGui.QMainWindow.__init__(self, parent)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        
        self.basic_parameter = dict()  #存储基本参数
        self.toolButton_pfdata.clicked.connect(self.toolButton_pfdata_c)
        self.toolButton__bsedata.clicked.connect(self.toolButton__bsedata_c)
        self.toolButton_changedata.clicked.connect(self.toolButton_changedata_c)
        self.toolButton_swntdata.clicked.connect(self.toolButton_swntdata_c)
        self.toolButton_faultratedata.clicked.connect(self.toolButton_faultratedata_c)
        self.toolButton_mapdata.clicked.connect(self.toolButton_mapdata_c)
        self.toolButton_swntprogram.clicked.connect(self.toolButton_swntprogram_c)
        self.toolButton_pfprogram.clicked.connect(self.toolButton_pfprogram_c)
        self.toolButton_resultfile.clicked.connect(self.toolButton_resultfile_c)
        self.pushButton_saveparameter.clicked.connect(self.saveparameter_c)
        self.pushButton_denyparameter.clicked.connect(self.denyparameter_c)
        self.matplotlibwidget_mapkeyfault.axes.set_axis_off()
        self.matplotlibwidget_mapsens.axes.set_axis_off()
        self.matplotlibwidget_barkeyfault.axes.set_axis_off()
        self.matplotlibwidget_barbus.axes.set_axis_off()
        self.matplotlibwidget_curveoptim.axes.set_axis_off()
        self.matplotlibwidget_mapallocation.axes.set_axis_off()

        self.pushButton_faultanalyse.clicked.connect(self.faultanalyse)
        self.pushButton_addfault.clicked.connect(self.add_fault)
        self.pushButton_kmeansfault.clicked.connect(self.kmeansfault)
        self.pushButton_mapkeyfault.clicked.connect(self.mapfault)
        self.pushButton_senscompute.clicked.connect(self.senscompute)
        self.pushButton_addbus.clicked.connect(self.add_bus)
        self.pushButton_senskmeans.clicked.connect(self.senskmeans)
        self.pushButton_mapsens.clicked.connect(self.mapsens)
        self.pushButton_optim.clicked.connect(self.optim_alpha)
        self.pushButton_mapallocation.clicked.connect(self.map_allocation)
        
    #文件路径选择函数        
    def toolButton_pfdata_c(self):
        #price = int(self.price_box.toPlainText())
        data_path = QtGui.QFileDialog.getOpenFileName(self, "Open File...", None,"DAT-Files (*.DAT *.dat);;All Files (*)")
        self.lineEdit_pfdata.setText(data_path)
    def toolButton__bsedata_c(self):
        data_path = QtGui.QFileDialog.getOpenFileName(self, "Open File...", None,"DAT-Files (*.bse *.BSE);;All Files (*)")
        self.lineEdit_bsedata.setText(data_path)
    def toolButton_changedata_c(self):
        data_path = QtGui.QFileDialog.getOpenFileName(self, "Open File...", None,"DAT-Files (*.DAT *.dat);;All Files (*)")
        self.lineEdit_changedata.setText(data_path)
    def toolButton_swntdata_c(self):
        data_path = QtGui.QFileDialog.getOpenFileName(self, "Open File...", None,"DAT-Files (*.swi *.SWI);;All Files (*)")
        self.lineEdit_swntdata.setText(data_path)
    def toolButton_faultratedata_c(self):
        data_path = QtGui.QFileDialog.getOpenFileName(self, "Open File...", None,"DAT-Files (*.TXT *.txt);;All Files (*)")
        self.lineEdit_faultratedata.setText(data_path)
    def toolButton_mapdata_c(self):
        data_path = QtGui.QFileDialog.getOpenFileName(self, "Open File...", None,"DAT-Files (*.TXT *.txt);;All Files (*)")
        self.lineEdit_mapdata.setText(data_path)
    def toolButton_swntprogram_c(self):
        data_path = QtGui.QFileDialog.getOpenFileName(self, "Open File...", None,"DAT-Files (*.exe *.EXE);;All Files (*)")
        self.lineEdit_swntprogram.setText(data_path)
    def toolButton_pfprogram_c(self):
        data_path = QtGui.QFileDialog.getOpenFileName(self, "Open File...", None,"DAT-Files (*.exe *.EXE);;All Files (*)")
        self.lineEdit_pfprogram.setText(data_path)
    def toolButton_resultfile_c(self):
        data_path = QtGui.QFileDialog.getExistingDirectory()
        self.lineEdit_resultfile.setText(data_path)
    
    def saveparameter_c(self):
        self.basic_parameter['pfdata'] = self.lineEdit_pfdata.text()
        self.basic_parameter['bsedata'] = self.lineEdit_bsedata.text()
        self.basic_parameter['changedata'] = self.lineEdit_changedata.text()
        self.basic_parameter['swntdata'] = self.lineEdit_swntdata.text()
        self.basic_parameter['faultratedata'] = self.lineEdit_faultratedata.text()
        self.basic_parameter['mapdata'] = self.lineEdit_mapdata.text()
        self.basic_parameter['swntprogram'] = self.lineEdit_swntprogram.text()
        self.basic_parameter['pfprogram'] = self.lineEdit_pfprogram.text()
        self.basic_parameter['resultfile'] = self.lineEdit_resultfile.text()
        self.basic_parameter['binary_statcom'] = bool(self.checkBox_statcom.text())
        self.basic_parameter['binary_svc'] = bool(self.checkBox_svc.text())
        self.basic_parameter['maxrate_sta'] = int(self.lineEdit__cc.text())
        self.basic_parameter['maxrate_svc'] = int(self.lineEdit_maxratesvc.text())
        self.basic_parameter['mingap_sta'] = int(self.lineEdit_mingapstatcom.text())
        self.basic_parameter['mingap_svc'] = int(self.lineEdit_mingapsvc.text())
        self.basic_parameter['fixedcost_sta'] = float(self.lineEdit_fixedcostatcom.text())
        self.basic_parameter['fixedcost_svc'] = float(self.lineEdit_fixedcostsvc.text())
        self.basic_parameter['pcost_sta'] = float(self.lineEdit_pcoststatcom.text())  
        self.basic_parameter['pcost_svc'] = float(self.lineEdit_pcostsvc.text())
        self.basic_parameter['loadrate'] = float(self.doubleSpinBox_loadrate.text())
        self.basic_parameter['dlrate'] = float(self.doubleSpinBox_dlrate.text())
        self.path_train_data = os.path.join(self.cwd,'train_data.txt')
        self.path_smp_key = os.path.join(self.cwd,'smp_key.txt')
        self.path_bus_potential = os.path.join(self.cwd,'bus_potential.txt') 
        
        self.path_dat = self.basic_parameter['pfdata'] 
        self.path_bse = self.basic_parameter['bsedata']
        self.path_dat_changes = self.basic_parameter['changedata']
        self.path_swi = self.basic_parameter['swntdata']
        self.path_fault_rate = self.basic_parameter['faultratedata']
        self.path_map_info = self.basic_parameter['mapdata']
        self.path_batbpa = self.basic_parameter['swntprogram']
        self.path_pfnt = self.basic_parameter['pfprogram']
        self.path_result = self.basic_parameter['resultfile']
        self.path_flt = os.path.join(self.cwd,'IEEE39.FLT')
        self.path_location_prob = os.path.join(self.cwd,'location_prob.txt')
        self.path_type_prob = os.path.join(self.cwd,'type_prob.txt')
        self.branch_prob = Sample.cal_branch(self.path_dat,self.path_fault_rate)
        self.load_prob = Sample.cal_loadlevel(1,1+self.basic_parameter['loadrate'])
        self.dynamicload_prob = Sample.cal_dynamicload(1,self.basic_parameter['dlrate'])
        self.location_prob = read_csv(self.path_location_prob, encoding='GBK', sep=' ', names=['vlevel','location','prob'], dtype=float, na_filter=False)
        self.type_prob = read_csv(self.path_type_prob, encoding='GBK', sep=' ', names=['vlevel','fault_type','prob'], dtype=float, na_filter=False)
        self.cleartime_prob  = Sample.cal_cleartime(1)  
        self.smp_whole = Sample.whole_space(self.load_prob, self.dynamicload_prob, self.branch_prob, self.type_prob,\
        self.location_prob, self.cleartime_prob) 
        
        self.map_info = read_csv(self.path_map_info, encoding='GBK', sep='	', names=['bus_name','x','y'], dtype=str, na_filter=False)
        self.x = self.map_info.x.apply(lambda i: float(i))
        self.y = self.map_info.y.apply(lambda i: float(i))

    def denyparameter_c(self):
        self.basic_parameter = dict()

    def faultanalyse(self):
 
        FormFile.bse_file(self.path_pfnt, self.path_dat_changes, self.load_prob)
        self.df_swi_dyld = FormFile.swi_file_dynamic_load(self.path_swi, self.dynamicload_prob)        
        
        self.smp, self.ind = KeyFault.find_first(self.smp_whole, self.path_flt, self.path_swi, self.path_bse, self.load_prob, self.dynamicload_prob,\
        self.smp_whole, self.branch_prob, self.path_result,self.path_batbpa)
        self.matplotlibwidget_barkeyfault.axes.bar(self.smp.index,self.ind['ind_smp'],align="center")
        self.matplotlibwidget_barkeyfault.axes.set_xlabel('Number of contingencies(smp_whole.index)')
        self.matplotlibwidget_barkeyfault.axes.set_ylabel('$f_1|_\phi=n$')
        
        self.matplotlibwidget_barkeyfault.axes.set_xticks(range(0,len(self.smp),1))
        self.matplotlibwidget_barkeyfault.axes.set_xticklabels(self.smp.index,fontsize = 4 )
        self.gap1 = list()
        self.gap1s = list()
        for i in range(10):
            self.gap1.append(0+i*max(self.ind['ind_smp'])*1.2/9)
            self.gap1s.append('%.1e'%(0+i*max(self.ind['ind_smp'])*1.2/9))
            
        self.matplotlibwidget_barkeyfault.axes.set_yticks(self.gap1)
        self.matplotlibwidget_barkeyfault.axes.set_yticklabels(self.gap1s, fontsize=4)
        
        self.matplotlibwidget_barkeyfault.axes.set_xlim(-1,len(self.smp)+1)
        self.matplotlibwidget_barkeyfault.draw()
        
        for i in range(len(self.smp.index)):
            self.comboBox_choosefault.addItem(str(self.smp.index[i]))
    def add_fault(self):
        self.listWidget_ignorefault.addItem(self.comboBox_choosefault.currentText())
        
    def kmeansfault(self):
        self.index_key, self.smp_key = KeyFault.find_next([31], self.ind, self.smp)
        ind_smp  = DataFrame(self.ind['ind_smp'])
        self.matplotlibwidget_barkeyfault.axes.bar(self.smp.index,ind_smp[0], label="Non-critical contingencies",align="center")
        self.matplotlibwidget_barkeyfault.axes.hold(True)
        self.matplotlibwidget_barkeyfault.axes.bar(self.index_key, ind_smp[0][self.index_key], color="red", label="Critical contingencies", align="center")
        self.matplotlibwidget_barkeyfault.axes.hold(True)        
        self.matplotlibwidget_barkeyfault.axes.bar([31], ind_smp[0][31], color="green", label="Ignorent contingencies", align="center")
        
        self.matplotlibwidget_barkeyfault.axes.legend(fontsize = 8, loc=2)

        self.matplotlibwidget_barkeyfault.axes.set_xlabel('Number of contingency branches')
        self.matplotlibwidget_barkeyfault.axes.set_ylabel('$f_1|_\phi=n$')
        self.matplotlibwidget_barkeyfault.axes.set_xticks(range(len(self.smp)))
        self.matplotlibwidget_barkeyfault.axes.set_xticklabels(self.smp.index,fontsize = 4 )
        self.matplotlibwidget_barkeyfault.axes.set_yticks(self.gap1)
        self.matplotlibwidget_barkeyfault.axes.set_yticklabels(self.gap1s,fontsize=4)
        self.matplotlibwidget_barkeyfault.axes.set_xlim(-1,len(self.smp)+1)
        self.matplotlibwidget_barkeyfault.draw()
    def mapfault(self):
#        self.map_info = read_csv(self.path_map_info, encoding='GBK', sep='	', names=['bus_name','x','y'], dtype=str, na_filter=False)
#        self.x = self.map_info.x.apply(lambda i: float(i))
#        self.y = self.map_info.y.apply(lambda i: float(i))


        for i in self.branch_prob.index:
            x1 = float(self.map_info.x[self.map_info.bus_name == self.branch_prob.from_bus[i]])
            y1 = float(self.map_info.y[self.map_info.bus_name == self.branch_prob.from_bus[i]])
            x2 = float(self.map_info.x[self.map_info.bus_name == self.branch_prob.to_bus[i]])
            y2 = float(self.map_info.y[self.map_info.bus_name == self.branch_prob.to_bus[i]])
            if i in self.index_key:
                self.matplotlibwidget_mapkeyfault.axes.plot([x1,x2],[y1,y2],'-r',linewidth=2)  
                self.matplotlibwidget_mapkeyfault.axes.hold(True)
            elif i == 31:
                self.matplotlibwidget_mapkeyfault.axes.plot([x1,x2],[y1,y2],'-g',linewidth=2)  
                self.matplotlibwidget_mapkeyfault.axes.hold(True)
            else:
                self.matplotlibwidget_mapkeyfault.axes.plot([x1,x2],[y1,y2],'-b',linewidth=2)  
                self.matplotlibwidget_mapkeyfault.axes.hold(True)    
        for i in self.map_info.index:
            self.matplotlibwidget_mapkeyfault.axes.text(self.x[i], self.y[i], self.map_info.bus_name[i], fontsize=8, va='bottom')    
        self.matplotlibwidget_mapkeyfault.axes.scatter(self.x,self.y,edgecolor='b',s=50)
        self.matplotlibwidget_mapkeyfault.axes.set_axis_off()
        self.matplotlibwidget_mapkeyfault.draw()
    def senscompute(self):
        FormFile.flt_file(self.path_flt, self.load_prob, self.dynamicload_prob, self.smp_key, self.branch_prob) 
        self.bus_info, self.sens = PotentialBus.find_first(0, 10, self.path_dat, self.path_swi, self.path_flt, self.path_bse,\
        self.path_result,self.path_batbpa, self.smp_key, self.load_prob, self.dynamicload_prob, self.df_swi_dyld)
        
        self.matplotlibwidget_barbus.axes.bar(self.bus_info.index,self.sens,align="center")
        self.matplotlibwidget_barbus.axes.set_xlabel('Number of bus')
        self.matplotlibwidget_barbus.axes.set_ylabel('$s_i$')
        
        self.matplotlibwidget_barbus.axes.set_xticks(range(len(self.bus_info)+1))
        self.matplotlibwidget_barbus.axes.set_xticklabels(self.bus_info.index+1,fontsize = 4 )
        self.gap2 = list()
        self.gap2s = list()
        for i in range(10):
            self.gap2.append((min(self.sens)-(max(self.sens)-min(self.sens))*0.2)+i*((max(self.sens)+(max(self.sens)-min(self.sens))*0.2)-\
            (min(self.sens)-(max(self.sens)-min(self.sens))*0.2))/9)
            self.gap2s.append('%.1e'%((min(self.sens)-(max(self.sens)-min(self.sens))*0.2)+i*((max(self.sens)+(max(self.sens)-min(self.sens))*0.2)-\
            (min(self.sens)-(max(self.sens)-min(self.sens))*0.2))/9))
            
        
        self.matplotlibwidget_barbus.axes.set_yticks(self.gap2)
        self.matplotlibwidget_barbus.axes.set_yticklabels(self.gap2s, fontsize=4)
        
        self.matplotlibwidget_barbus.axes.set_xlim(-1,len(self.bus_info)+1)
        self.matplotlibwidget_barbus.axes.set_ylim(min(self.sens)-(max(self.sens)-min(self.sens))*0.2,max(self.sens)\
        +(max(self.sens)-min(self.sens))*0.2)
        self.matplotlibwidget_barbus.draw()
        for i in range(len(self.bus_info.index)):
           self.comboBox_choosebus.addItem(str(self.bus_info.index[i]))
    def add_bus(self):
        self.listWidget_ignorebus.addItem(self.comboBox_choosebus.currentText())
    def senskmeans(self):
        self.index_potential, self.bus_potential = PotentialBus.find_next(self.bus_info, self.sens, [21])
        self.sens  = DataFrame(self.sens)
        self.matplotlibwidget_barbus.axes.bar(self.bus_info.index+1,self.sens[0], label="Non-sensitive buses",align="center")
        self.matplotlibwidget_barbus.axes.hold(True)      
        self.matplotlibwidget_barbus.axes.set_xlabel('Number of buses')
        self.matplotlibwidget_barbus.axes.set_ylabel('$s_i$')
        self.matplotlibwidget_barbus.axes.set_xticks(range(len(self.bus_info)+1))
        self.matplotlibwidget_barbus.axes.set_xticklabels(self.bus_info.index+1,fontsize = 4 )
 
        self.matplotlibwidget_barbus.axes.set_yticks(self.gap2)
        self.matplotlibwidget_barbus.axes.set_yticklabels(self.gap2s, fontsize=4)
        
        self.matplotlibwidget_barbus.axes.set_xlim(-1,len(self.bus_info)+1)
        self.matplotlibwidget_barbus.axes.set_ylim(min(self.sens[0])-(max(self.sens[0])-min(self.sens[0]))*0.2,max(self.sens[0])\
        +(max(self.sens[0])-min(self.sens[0]))*0.2)
        a =list()
        for i in self.index_potential:
            a.append(i+1)
        self.matplotlibwidget_barbus.axes.bar(a, self.sens[0][self.index_potential], color="red", label="Sensitive buses", align="center")
        self.matplotlibwidget_barbus.axes.legend(fontsize = 8, loc=2)
        self.matplotlibwidget_barbus.draw()
    def mapsens(self):
        for i in self.branch_prob.index:
            x1 = float(self.map_info.x[self.map_info.bus_name == self.branch_prob.from_bus[i]])
            y1 = float(self.map_info.y[self.map_info.bus_name == self.branch_prob.from_bus[i]])
            x2 = float(self.map_info.x[self.map_info.bus_name == self.branch_prob.to_bus[i]])
            y2 = float(self.map_info.y[self.map_info.bus_name == self.branch_prob.to_bus[i]])
            if i in self.index_key:
                self.matplotlibwidget_mapsens.axes.plot([x1,x2],[y1,y2],'-r',linewidth=2)  
                self.matplotlibwidget_mapsens.axes.hold(True)
            elif i == 31:
                self.matplotlibwidget_mapsens.axes.plot([x1,x2],[y1,y2],'-g',linewidth=2)  
                self.matplotlibwidget_mapsens.axes.hold(True)
            else:
                self.matplotlibwidget_mapsens.axes.plot([x1,x2],[y1,y2],'-b',linewidth=2)  
                self.matplotlibwidget_mapsens.axes.hold(True)    
        for i in self.map_info.index:
            self.matplotlibwidget_mapsens.axes.text(self.x[i], self.y[i], self.map_info.bus_name[i], fontsize=8, va='bottom')   
            
            if i in self.index_potential:
                self.matplotlibwidget_mapsens.axes.scatter(self.x[i],self.y[i],c='r',s=150)
                self.matplotlibwidget_mapsens.axes.hold(True)
            else:
                self.matplotlibwidget_mapsens.axes.scatter(self.x[i],self.y[i],c='b',s=50)
                self.matplotlibwidget_mapsens.axes.hold(True)
        self.matplotlibwidget_mapsens.axes.set_axis_off()
        self.matplotlibwidget_mapsens.draw()
        
    def optim_alpha(self):
        self.X_optimal, self.y1, self.y2 = optimization_alpha(self.path_train_data,self.path_smp_key,self.path_bus_potential,[],[],\
        self.basic_parameter['maxrate_sta'], self.basic_parameter['mingap_sta'], self.basic_parameter['fixedcost_sta'],\
        self.basic_parameter['pcost_sta'], self.basic_parameter['fixedcost_svc'],self.basic_parameter['pcost_svc'],\
        int(self.spinBox_randomseed.text()), int(self.spinBox_popsize.text()),int(self.spinBox_maxinterate.text()))
        self.matplotlibwidget_curveoptim.axes.scatter(self.y1,self.y2)
        self.matplotlibwidget_curveoptim.axes.set_xlim([min(self.y1)*0.95,max(self.y1)*1.05])
        self.matplotlibwidget_curveoptim.axes.set_ylim([min(self.y2)*0.95,max(self.y2)*1.05])
        self.matplotlibwidget_curveoptim.axes.set_xlabel('Cost/(10 thousand yuan)',fontsize=8)
        self.matplotlibwidget_curveoptim.axes.set_ylabel('Short-term voltage instability risk',fontsize=8)
        
        self.matplotlibwidget_curveoptim.draw()
        
        for i in range(len(self.y1)):
            self.comboBox_objective.addItem('Cost='+str(self.y1[i])+'\t'+'STVISR='+str(self.y2[i]))
    def map_allocation(self):
        self.matplotlibwidget_mapallocation.update()
        for i in self.branch_prob.index:
            x1 = float(self.map_info.x[self.map_info.bus_name == self.branch_prob.from_bus[i]])
            y1 = float(self.map_info.y[self.map_info.bus_name == self.branch_prob.from_bus[i]])
            x2 = float(self.map_info.x[self.map_info.bus_name == self.branch_prob.to_bus[i]])
            y2 = float(self.map_info.y[self.map_info.bus_name == self.branch_prob.to_bus[i]])
            if i in self.index_key:
                self.matplotlibwidget_mapallocation.axes.plot([x1,x2],[y1,y2],'-r',linewidth=2)  
                self.matplotlibwidget_mapallocation.axes.hold(True)
            elif i == 31:
                self.matplotlibwidget_mapallocation.axes.plot([x1,x2],[y1,y2],'-g',linewidth=2)  
                self.matplotlibwidget_mapallocation.axes.hold(True)
            else:
                self.matplotlibwidget_mapallocation.axes.plot([x1,x2],[y1,y2],'-b',linewidth=2)  
                self.matplotlibwidget_mapallocation.axes.hold(True)    
        for i in self.map_info.index:
            self.matplotlibwidget_mapallocation.axes.text(self.x[i], self.y[i], self.map_info.bus_name[i], fontsize=8, va='bottom')   
            
            if i in self.index_potential:
                self.matplotlibwidget_mapallocation.axes.scatter(self.x[i],self.y[i],c='r',s=150)
                self.matplotlibwidget_mapallocation.axes.hold(True)
            else:
                self.matplotlibwidget_mapallocation.axes.scatter(self.x[i],self.y[i],c='b',s=50)
                self.matplotlibwidget_mapallocation.axes.hold(True)
        choose_index = self.comboBox_objective.currentIndex()
        allocation = self.X_optimal[choose_index]
        for i in range(0,len(allocation),2):
            if allocation[i]!=0 and allocation[i+1]==0:
                self.matplotlibwidget_mapallocation.axes.scatter(self.x[self.index_potential[i/2]],self.y[self.index_potential[i/2]],marker='^',c='y',s=150)
                self.matplotlibwidget_mapallocation.axes.hold(True)
                self.matplotlibwidget_mapallocation.axes.text(self.x[self.index_potential[i/2]],self.y[self.index_potential[i/2]], \
                str(allocation[i])+'MVar', fontsize=8, va='top')  
            elif allocation[i]==0 and allocation[i+1]!=0:
                self.matplotlibwidget_mapallocation.axes.scatter(self.x[self.index_potential[i/2]],self.y[self.index_potential[i/2]],marker='s',c='y',s=150)
                self.matplotlibwidget_mapallocation.axes.hold(True)
                self.matplotlibwidget_mapallocation.axes.text(self.x[self.index_potential[i/2]],self.y[self.index_potential[i/2]], \
                str(allocation[i+1])+'MVar', fontsize=8, va='top')  
  
        self.matplotlibwidget_mapallocation.axes.set_axis_off()
        self.matplotlibwidget_mapallocation.draw()       
        
        
        
        

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = DVP_App()
    
    window.show()  
    sys.exit(app.exec_())
