# library import
import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%% data import
PV_20 = pd.read_excel('20kW_PV.xlsx')
# print(PV_20['1월'])
PV_20 = np.array(PV_20)
PV_80 = pd.read_excel('80kW_PV.xlsx')
PV_80 = np.array(PV_80)
WT = pd.read_excel('WT.xlsx')
WT = np.array(WT)
Load = pd.read_excel('Load.xlsx')
Load = np.array(Load)
# print(month[0])
#%% 1월
PV_20 = PV_20[:,12]
PV_80 = PV_80[:,12]
WT = WT[:,12]
Load = Load[:,2]
#%% Variable 
# "output k-th segment unit j time step i" r[j][k[][i]:(3,10,24) / "output of unit j at time step i" : p[j][i]:(3,24)
r, p = [], []
for i in range(3):
    r.append([])
    p.append([])
    p[i] = cp.Variable(len(Load))
    for j in range(10):
        r[i].append(cp.Variable(len(Load)))
        
# u: on/off status at time step i of unit j / y : startup status at " / z : shutdown status at "
u, y, z = [], [], []
for i in range(3):
    u.append(cp.Variable(len(Load), integer=True))
    y.append(cp.Variable(len(Load), integer=True))
    z.append(cp.Variable(len(Load), integer=True))
    
# BESS charge and discharge at time step i
Uc = cp.Variable(len(Load), integer=True)
Ud = cp.Variable(len(Load), integer=True)
# BESS 
BESS_char = cp.Variable(len(Load))
BESS_dischar = cp.Variable(len(Load))
BESS = cp.Variable(len(Load))

SOC = cp.Variable(len(Load))

B_Rating = 500 # kW
B_capacity = 1000 # kWh
C_rate = 90/100
D_rate = 100/90
Time = 1
cost = 0
Start = 15000
Shut = 11000
#%% Piecewise linearization
# cost function
def cost1(p):
    return 1200*(0.001*p**2 + 0.15*p + 5.9)
def cost2(p):
    return 1200*(0.0011*p**2 + 0.145*p + 5.85)
def cost3(p):
    return 1200*(0.001*p**2 + 0.14*p + 5.95)
# each slope at k-th segment
sl1, sl2, sl3 = [], [], []
for i in range(10):
    sl1.append((cost1(7*(i+1)) - cost1(7*i)) / 7)
    sl2.append((cost2(7*(i+1)) - cost2(7*i)) / 7)
    sl3.append((cost3(7*(i+1)) - cost3(7*i)) / 7)
# mincost    
mincost1 = cost1(0)
mincost2 = cost2(0)
mincost3 = cost3(0)
#%% constraints
constraints = []
for i in range(len(Load)):
    ext = []
    # Load balance
    ext.append(Load[i] == p[0][i] + p[1][i] + p[2][i] + PV_20[i] + PV_80[i] + WT[i] + BESS[i])
    
    # BESS
    ext.append(Uc[i] >= 0) 
    ext.append(Uc[i] <= 1)
    ext.append(Ud[i] >= 0)
    ext.append(Ud[i] <= 1)
    ext.append(Uc[i] + Ud[i] <= 1)
            
    ext.append(BESS_char[i] >= -1.0 * (B_Rating * Uc[i]))
    ext.append(BESS_char[i] <= 0)
    ext.append(BESS_dischar[i] <= B_Rating * Ud[i])
    ext.append(BESS_dischar[i] >= 0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    
    ext.append(BESS[i] == BESS_char[i] + BESS_dischar[i])
    
    # SOC
    ext.append(SOC[i] >= 20)
    ext.append(SOC[i] <= 80)
    ext.append(SOC[i] == SOC[i-1] - ((BESS_char[i] * 100 * C_rate) / (B_capacity * Time)) - ((BESS_dischar[i] * 100 * D_rate) / (B_capacity * Time))
               if i >= 1 else SOC[i] == 50)
    ext.append(SOC[23] == 50)
    
    for j in range(3):
        # Generation capacity
        ext.append(p[j][i] >= 10 * u[j][i])
        ext.append(p[j][i] <= 80 * u[j][i])
        
        ext.append(p[j][i] == (10 * u[j][i] + r[j][0][i] + r[j][1][i] + r[j][2][i] + r[j][3][i] + r[j][4][i] + r[j][5][i] +
                   r[j][6][i] + r[j][7][i] + r[j][8][i] + r[j][9][i]))
        
        # integer
        ext.append(u[j][i] >= 0)
        ext.append(u[j][i] <= 1)
        ext.append(y[j][i] >= 0)
        ext.append(y[j][i] <= 1)
        ext.append(z[j][i] >= 0)
        ext.append(z[j][i] <= 1)
        ext.append(u[j][i] - u[j][i-1] == y[j][i] - z[j][i] if i >= 1 else u[j][0] == y[j][0] - z[j][0])
        ext.append(y[j][i] + z[j][i] <= 1)
        
        # segment
        for k in range(10):
            ext.append(r[j][k][i] >= 0)
            ext.append(r[j][k][i] <= 7 * u[j][i])
    # sum every time
    constraints += ext
#%% objective function
obj = 0
for i in range(24):
    obj += mincost1 * u[0][i] + mincost2 * u[1][i] + mincost3 * u[2][i]
    obj += Start * y[0][i] + Shut * z[0][i] + Start * y[1][i] + Shut * z[1][i] + Start * y[2][i] + Shut * z[2][i]
    for k in range(10):
        obj += sl1[k] * r[0][k][i] + sl2[k] * r[1][k][i] + sl3[k] * r[2][k][i]
#%% Optimal Schedule
prob = cp.Problem(cp.Minimize(obj), constraints)
prob.solve(solver=cp.MOSEK)
a = np.round(prob.value, 2)
print('cost :', a, 'KRW')
print(prob.status)
#%% Data for Plot
SOV_val = SOC.value
p_gen1 = p[0].value
p_gen2 = p[1].value
p_gen3 = p[2].value
BESS_char_val = BESS_char.value
BESS_dischar_val = BESS_dischar.value
BESS_val = BESS.value
POWER = PV_20 + PV_80 + WT
# print(p.value)

#%% Plot
Fig = plt.figure(figsize=[15,10])
ax1 = plt.subplot(2, 1, 2)
plt.xlabel('time (h)')
plt.ylabel('Output (MW)')
plt.suptitle("MILP UC with 3 Gen in Dec", fontsize=20)
plt.plot(Load, 'g-', label = 'Demand')
plt.plot(p[0].value, 'r.-', label = 'Gen1')
plt.plot(p[1].value, 'c-', label = 'Gen2')
plt.plot(p[2].value, 'b-', label = 'Gen3')
plt.plot(BESS_val, 'y', label = 'BESS')
plt.plot(POWER, 'm', label = 'PV, WT')
plt.legend(loc=2)

plt.subplot(2, 1, 1, sharex=ax1)
plt.ylabel('SOC of BESS (%)')
plt.plot(SOC.value, 'y', label = 'SOC')
plt.xticks(visible=False)
plt.legend(loc=2)
# plt.plot(p_gen1 + p_gen2 + p_gen3 + PV_20 + PV_80 + WT + BESS_val, 'k.-') # 일치
plt.show

# for i in range(24):
    # print(SOC.value)
    # print(Uc[i].value, Ud[i].value, BESS_char[i].value, BESS_dischar[i].value)