# library import
import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import gurobipy
# %% data import
PV_200 = pd.read_excel('200kW_PV.xlsx')
# print(PV_200['1월'])
PV_200 = np.array(PV_200)
# PV_200 = PV_20 * 10
PV_400 = pd.read_excel('600kW_PV.xlsx')
PV_400 = np.array(PV_400)
# PV_400 = PV_80 * 5
WT = pd.read_excel('WT.xlsx')
WT = np.array(WT)
Load = pd.read_excel('Load.xlsx')
Load = np.array(Load)
# print(month[0])
# %% 1월
PV_200 = PV_200[:, 1]
PV_400 = PV_400[:, 1]
WT = WT[:, 1]
Load = Load[:, 2]
#%% Parameter
GenNum = 3
PVNum = 2
T = len(Load)
Piece = 10
MinGen = 15  # kW (30% of Max)
B_Rating = 500  # kW
B_capacity = 1000  # kWh
C_rate = 90/100
D_rate = 100/90
Time = 1
cost = 0
Start = 15000
Shut = 11000
# %% Variable
# ---"output k-th segment unit j time step i" r[j][k][i]:(3,10,24) / "output of unit j at time step i" : p[j][i]:(3,24)---
r, p = [], []
for i in range(GenNum):
    r.append([])
    p.append([])
    p[i] = cp.Variable(T)
    for j in range(Piece):
        r[i].append(cp.Variable(T))
# ---u: on/off status at time step i of unit j / y : startup status at " / z : shutdown status at "---
u, y, z = [], [], []
for i in range(GenNum):
    u.append(cp.Variable(T, integer=True))
    y.append(cp.Variable(T, integer=True))
    z.append(cp.Variable(T, integer=True))
# ---ESS charge and discharge at time step i---
Uc = cp.Variable(T, integer=True)
Ud = cp.Variable(T, integer=True)
# ---ESS---
ESS_char = cp.Variable(T)
ESS_dischar = cp.Variable(T)
ESS = cp.Variable(T)
# ---SOC----
SOC = cp.Variable(T)
# ---PV Curtail, P_Curt[j][k][i]:(2,10,24)
P_Curt = []
for i in range(PVNum):
    P_Curt.append([])
    P_Curt[i] = cp.Variable(T)
curt = []
for i in range(PVNum):
    curt.append([])
    for j in range(Piece):
        curt[i].append(cp.Variable(T))
# ---PV Curtail on/off---
Upvc = []
for i in range(PVNum):
    Upvc.append([])
    Upvc[i] = cp.Variable(T, integer=True)
# ---Output---
PV_400_Output = cp.Variable(T)
PV_200_Output = cp.Variable(T)
# ---Virtual cost
s10 = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 10e-5]
# %% Piecewise linearization
# ---Cost function---
def cost1(p):
    return 1200*(0.001*p**2 + 0.15*p + 5.9)
def cost2(p):
    return 1200*(0.0011*p**2 + 0.145*p + 5.85)
def cost3(p):
    return 1200*(0.001*p**2 + 0.14*p + 5.95)
# ---each slope at k-th segment---
sl1, sl2, sl3 = [], [], []
for i in range(Piece):
    sl1.append((cost1(7*(i+1)) - cost1(7*i)) / 7)
    sl2.append((cost2(7*(i+1)) - cost2(7*i)) / 7)
    sl3.append((cost3(7*(i+1)) - cost3(7*i)) / 7)
# mincost
mincost1 = cost1(0)
mincost2 = cost2(0)
mincost3 = cost3(0)
# %% Constraints
constraints = []
for i in range(T):
    ext = []
# ---Power balance---
    ext.append(Load[i] == p[0][i] + p[1][i] + p[2][i] + PV_200[i] +
               PV_400[i] + WT[i] + ESS[i] - P_Curt[0][i] - P_Curt[1][i])

# ---ESS---
    ext.append(Uc[i] >= 0)
    ext.append(Uc[i] <= 1)
    ext.append(Ud[i] >= 0)
    ext.append(Ud[i] <= 1)
    ext.append(Uc[i] + Ud[i] <= 1)

    ext.append(ESS_char[i] >= -1.0 * (B_Rating * Uc[i]))
    ext.append(ESS_char[i] <= 0)
    ext.append(ESS_dischar[i] <= B_Rating * Ud[i])
    ext.append(ESS_dischar[i] >= 0)

    ext.append(ESS[i] == ESS_char[i] + ESS_dischar[i])
# ---SOC---
    ext.append(SOC[i] >= 20)
    ext.append(SOC[i] <= 80)
    ext.append(SOC[i] == SOC[i-1] - ((ESS_char[i] * 100 * C_rate) / (B_capacity * Time))
               - ((ESS_dischar[i] * 100 * D_rate) / (B_capacity * Time))
               if i >= 1 else SOC[i] == 50)
    ext.append(SOC[23] == 50)

# ---Generation capacity---
    for j in range(GenNum):
        ext.append(p[j][i] >= MinGen) # Always On
        ext.append(p[j][i] <= 80 * u[j][i])
        ext.append(p[j][i] == (10 * u[j][i] + r[j][0][i] + r[j][1][i] + r[j][2][i] + r[j][3][i] + r[j][4][i]
                               + r[j][5][i] + r[j][6][i] + r[j][7][i] + r[j][8][i] + r[j][9][i]))
# ---Integer---
        ext.append(u[j][i] >= 0)
        ext.append(u[j][i] <= 1)
        ext.append(y[j][i] >= 0)
        ext.append(y[j][i] <= 1)
        ext.append(z[j][i] >= 0)
        ext.append(z[j][i] <= 1)
        ext.append(u[j][i] - u[j][i-1] == y[j][i] - z[j][i] if i >= 1 else u[j][0] == y[j][0] - z[j][0])
        ext.append(y[j][i] + z[j][i] <= 1)
# ---Segment---
        for k in range(Piece):
            ext.append(r[j][k][i] >= 0)
            ext.append(r[j][k][i] <= 7 * u[j][i])
# ---Curtailment---
    for j in range(PVNum):
        ext.append(Upvc[j][i] >= 0)
        ext.append(Upvc[j][i] <= 1)

    ext.append(P_Curt[0][i] >= 0)
    ext.append(P_Curt[0][i] <= PV_400[i] * Upvc[0][i])
    ext.append(P_Curt[1][i] >= 0)
    ext.append(P_Curt[1][i] <= PV_200[i] * Upvc[1][i])

    for k in range(Piece):
        ext.append(curt[0][k][i] >= 0)
        ext.append(curt[0][k][i] <= 60/Piece)
    for k in range(Piece):
        ext.append(curt[1][k][i] >= 0)
        ext.append(curt[1][k][i] <= 30/Piece)

    ext.append(P_Curt[0][i] == curt[0][0][i] + curt[0][1][i] + curt[0][2][i] + curt[0][3][i] + curt[0][4][i] + curt[0][5][i] +
               curt[0][6][i] + curt[0][7][i] + curt[0][8][i] + curt[0][9][i])
    ext.append(P_Curt[1][i] == curt[1][0][i] + curt[1][1][i] + curt[1][2][i] + curt[1][3][i] + curt[1][4][i] + curt[1][5][i] +
               curt[1][6][i] + curt[1][7][i] + curt[1][8][i] + curt[1][9][i])
 
    ext.append(PV_400_Output[i] == PV_400[i] - P_Curt[0][i])
    ext.append(PV_200_Output[i] == PV_200[i] - P_Curt[1][i])
# ---Sum every time---
    constraints += ext
# %% Objective function
obj = 0
for i in range(24):
    obj += mincost1 * u[0][i] + mincost2 * u[1][i] + mincost3 * u[2][i]
    # obj += Start * y[0][i] + Shut * z[0][i] + Start * y[1][i] + Shut * z[1][i] + Start * y[2][i] + Shut * z[2][i] # ED Always ON
    for j in range(Piece):
        obj += sl1[j] * r[0][j][i] + sl2[j] * r[1][j][i] + sl3[j] * r[2][j][i] + s10[j] * curt[0][j][i] + s10[j] * curt[1][j][i]
        
# %% Optimal Schedule
prob = cp.Problem(cp.Minimize(obj), constraints)
prob.solve(solver=cp.MOSEK)
a = np.round(prob.value, 2)
print('cost :', a, 'KRW')
print(prob.status)
# %% Data for Plot
SOV_val = SOC.value
p_gen1 = p[0].value
p_gen2 = p[1].value
p_gen3 = p[2].value
ESS_char_val = ESS_char.value
ESS_dischar_val = ESS_dischar.value
ESS_val = ESS.value
# %% Plot
# ---Total---
Fig = plt.figure(figsize=[15, 10])
ax1 = plt.subplot(2, 1, 2)
plt.xlabel('time (h)')
plt.ylabel('Output (kW)')
plt.suptitle("3Gen MILP ED Curtailment", fontsize=25)
plt.plot(Load, color='brown', label='Load')
plt.plot(p[0].value, color='cyan', marker=11, label='Diesel1')
plt.plot(p[1].value, color='red', marker=10, label='Diesel2')
plt.plot(p[2].value, color='blue', label='Diesel3')
plt.plot(ESS_val, color='green', label='ESS')
plt.plot(PV_400_Output.value, 'olive', label='PV_400')
plt.plot(PV_200_Output.value, 'orange', label='PV_200')
plt.legend(loc=2)
plt.subplot(2, 1, 1, sharex=ax1)
plt.ylabel('SOC of ESS (%)')
plt.plot(SOC.value, color='green', label='SOC')
plt.xticks(visible=False)
plt.legend(loc=2)
# plt.plot(p_gen1 + p_gen2 + p_gen3 + PV_200 + PV_400 + WT + ESS_val, 'k.-') # 일치
plt.show
# ---PV1 Curtail---
Fig = plt.figure(figsize=[15,10])
plt.xlabel('time (h)')
plt.ylabel('Output (kW)')
plt.suptitle("Curtailment 600kW distributed", fontsize=25)
plt.plot(PV_400, color='green', label='PV_600')
plt.plot(P_Curt[0].value, color='brown', label='Curtail')
plt.plot(PV_400_Output.value, color='blue', label='PV_600 Output')
plt.legend(loc=2)
plt.show
# ---PV2 Curtail
Fig = plt.figure(figsize=[15,10])
plt.xlabel('time (h)')
plt.ylabel('Output (kW)')
plt.suptitle("Curtailment 200kW distributed", fontsize=25)
plt.plot(PV_200, color='green', label='PV_200')
plt.plot(P_Curt[1].value, color='brown', label='Curtail')
plt.plot(PV_200_Output.value, color='blue', label='PV_200 Output')
plt.legend(loc=2)
plt.show

# print(PV_400)
print(P_Curt[0].value)
# print(PV_400_Output.value)
# print(PV_200)
print(P_Curt[1].value)
# print(PV_200_Output.value)