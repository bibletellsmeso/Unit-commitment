#!/usr/bin/env python
# coding: utf-8

import pulp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

'''PV Data from KPX'''

signal = pd.read_csv('KPX_PV.csv', sep=',', names=['Source', 'Location', 'Date', 'Hour', 'Power'], encoding='CP949')[1:]
signal = pd.DataFrame(signal,columns=['Hour', 'Power']).to_numpy(dtype=np.float32)

for i in range(24):
    globals()['signal_{}'.format(i)] = signal[signal[:,0] == i, -1]

mean = np.array([])
var = np.array([])

for i in range(24):
    mean = np.append(mean, np.nanmean(globals()['signal_{}'.format(i)]))
    var = np.append(var, np.nanvar(globals()['signal_{}'.format(i)]))

UB = mean+1.96*np.sqrt(var)
LB = mean-1.96*np.sqrt(var); LB[LB < 0] = 0

print("Mean : {}\nVar : {}".format(mean, var))

'''WT Data from KPX'''

signal_WT = pd.read_csv('KPX_WT.csv', sep=',', names=['Date', 'Hour', 'Location', 'Power'], encoding='CP949')[1:]
signal_WT = pd.DataFrame(signal_WT,columns=['Hour', 'Power']).to_numpy(dtype=np.float32)

for i in range(24):
    globals()['signal_WT_{}'.format(i)] = signal_WT[signal_WT[:,0] == i, -1]

mean_WT = np.array([])
var_WT = np.array([])

for i in range(24):
    mean_WT = np.append(mean_WT, np.nanmean(globals()['signal_WT_{}'.format(i)]))
    var_WT = np.append(var_WT, np.nanvar(globals()['signal_WT_{}'.format(i)]))

UB_WT = mean_WT+1.96*np.sqrt(var_WT)
LB_WT = mean_WT-1.96*np.sqrt(var_WT); LB_WT[LB_WT < 0] = 0

print("Mean : {}\nVar : {}".format(mean_WT, var_WT))

'''Load Data from KPX'''

signal_Load = pd.read_csv('KPX_Load.csv', sep=',', names=['Date', 'signal_Load_1', 'signal_Load_2', 'signal_Load_3'
                                                        , 'signal_Load_4', 'signal_Load_5', 'signal_Load_6'
                                                        , 'signal_Load_7', 'signal_Load_8', 'signal_Load_9'
                                                        , 'signal_Load_10', 'signal_Load_11', 'signal_Load_12'
                                                        , 'signal_Load_13', 'signal_Load_14', 'signal_Load_15'
                                                        , 'signal_Load_16', 'signal_Load_17', 'signal_Load_18'
                                                        , 'signal_Load_19', 'signal_Load_20', 'signal_Load_21'
                                                        , 'signal_Load_22', 'signal_Load_23', 'signal_Load_0'])[1:]
signal_Load = signal_Load.drop(['Date'], axis=1).to_numpy(dtype=np.float32)/1000

mean_Load = np.array([])
var_Load = np.array([])

for i in range(24):
    mean_Load = np.append(mean_Load, np.nanmean(signal_Load[:,i-1]))
    var_Load = np.append(var_Load, np.nanvar(signal_Load[:,i-1]))

UB_Load = mean_Load+1.96*np.sqrt(var_Load)
LB_Load = mean_Load-1.96*np.sqrt(var_Load)

print("Mean : {}\nVar : {}".format(mean_Load, var_Load))

'''Plot for Data'''

plt.figure(1)
for i in range(24):
    pv_max = max(signal[:,-1])
    pv_min = min(signal[:,-1])
    rv = stats.norm(loc=mean[i], scale=var[i])
    rv_range = np.arange(pv_min, pv_max, (pv_max - pv_min) / 50)
    pdf = rv.pdf(rv_range); #pdf[pdf > 1.0] = 1.0
    plt.subplot(6,4,i+1)
    plt.plot(rv_range,pdf)
    plt.vlines(rv_range, 0, pdf, colors='b', lw=5, alpha=0.5)
    #plt.ylim(0, 1.0)  # y축 범위

plt.figure(2)
for i in range(24):
    WT_max = max(signal_WT[:,-1])
    WT_min = min(signal_WT[:,-1])
    rv = stats.norm(loc=mean_WT[i], scale=var_WT[i])
    rv_range = np.arange(WT_min,WT_max,(WT_max - WT_min)/50)
    pdf = rv.pdf(rv_range)
    plt.subplot(6,4,i+1)
    plt.plot(rv_range,pdf)
    plt.vlines(rv_range, 0, pdf, colors='b', lw=5, alpha=0.5)
    #plt.ylim(0, 1.0)  # y축 범위

plt.figure(3)
for i in range(24):
    Load_max = max(signal_Load.flatten())
    Load_min = min(signal_Load.flatten())
    rv = stats.norm(loc=mean_Load[i], scale=var_Load[i])
    rv_range = np.arange(Load_min,Load_max,(Load_max - Load_min)/50)
    pdf = rv.pdf(rv_range)
    plt.subplot(6,4,i+1)
    plt.plot(rv_range,pdf)
    plt.vlines(rv_range, 0, pdf, colors='b', lw=5, alpha=0.5)
    #plt.ylim(0, 1.0)  # y축 범위


Fig = plt.figure(4, figsize=[15,10])
ax1 = plt.subplot(3,1,3)
plt.xlabel('Time(h)')
plt.suptitle("KPX Data", fontsize=20)
plt.subplot(311);
plt.plot(mean,'b',label='Mean')
plt.ylabel('PV Output(kW)')
plt.plot(UB,'g',linestyle='--',label='Bound')
plt.plot(LB,'g',linestyle='--',label='Bound')
plt.legend(['Mean', 'Bound'],loc='upper left')

plt.subplot(312)
plt.ylabel('WT Output(kW)')
plt.plot(mean_WT,'b',label='Mean')
plt.plot(UB_WT,'g',linestyle='--',label='Bound')
plt.plot(LB_WT,'g',linestyle='--',label='Bound')
plt.legend(['Mean', 'Bound'],loc='upper left')

plt.subplot(313)
plt.ylabel('Load Output(kW)')
plt.plot(mean_Load,'b',label='Mean')
plt.plot(UB_Load,'g',linestyle='--',label='Bound')
plt.plot(LB_Load,'g',linestyle='--',label='Bound')
plt.legend(['Mean', 'Bound'],loc='upper left')
plt.show()