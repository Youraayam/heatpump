# heatpump
energy analysis of heat pump
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 12:31:07 2023

@author: aayamc
"""


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
import plotly.offline as pyo


ass3 = pd.read_csv('ass3.csv',encoding='latin1')
ass3.info()
ass3.head()
ass3["Time"] = pd.to_datetime(ass3["Time"])
print(ass3.columns.tolist())
Time = ass3['Time']


#Evaporator
gly_in = ass3['TT04']
gly_out = ass3['TT05']
evp_in = ass3['RT02']
evp_out = ass3['RT06']
gly_pump = ass3['Glycol pump %']
CO2_comp = ass3['CO2 compressor % ']

plt.figure(figsize=(12,8))

plt.plot(Time,gly_in,label='Inlet temperature of glycol')
plt.plot(Time,gly_out, label ='Outlet temperatur of glycol')

 
# giving labels to the axises
plt.xlabel('Time(Day|Hour|minute)',fontsize=20, color = 'r')
plt.ylabel("Temperature(℃)",fontsize =20, color = 'g')
plt.title('Inlet and Outlet temperature of glycol',fontsize =20)
 
# defining display layout 
plt.tight_layout()
plt.grid() 
plt.legend()

plt.show()


plt.figure(figsize=(12,8))
plt.plot(Time,evp_in,label='CO2 inlet temperature at evaporator')
plt.plot(Time,evp_out, label ='CO2 outlet temperature at evaporator')
plt.xlabel('Time(Day|Hour|minute)',fontsize=20, color = 'r')
plt.ylabel("Temperature(℃)",fontsize =20, color = 'g')
 
plt.title('Inlet and Outlet temperature of CO2  of evaporator',fontsize =20)

plt.tight_layout()
plt.grid() 
plt.legend()

plt.show()

plt.figure(figsize=(12,8))
plt.plot(Time,abs(evp_in-gly_in),label='Temperature difference between inlet temperature of glycol and evaporator')
plt.plot(Time,abs(evp_out-gly_out), label ='Temperature difference between outlet temperature of glycol and evaporator')
plt.xlabel('Time(Day|Hour|minute)',fontsize=20, color = 'r')
plt.ylabel("Temperature(℃)",fontsize =20, color = 'g')

plt.title('Temperature difference between inlet temperature of glycol and evaporator and difference between outlet temperature of glycol and evaporator',fontsize =20)
 

plt.tight_layout()
plt.grid() 
plt.legend()

plt.show()

#Liquid_separator

liq_in = ass3['RT06']
liq_out = ass3['RT05']

plt.figure(figsize=(12,8))
plt.plot(Time,liq_in,label='Inlet temperature of liquid separator')
plt.plot(Time,liq_out, label ='Outlet temperature of liquid separator')
plt.xlabel('Time(Day|Hour|minute)',fontsize=20, color = 'r')
plt.ylabel("Temperature(℃)",fontsize =20, color = 'g')
 
plt.title('Inlet and Outlet temperature of Liquid separator',fontsize =20)

plt.tight_layout()
plt.grid() 
plt.legend()

plt.show()


#Internal_heat exchangers

Ihe_in = ass3['RT05']
Ihe_out = ass3['RT03']

plt.figure(figsize=(12,8))
plt.plot(Time,Ihe_in,label='Inlet temperature of internal heat exchanger')
plt.plot(Time,Ihe_out, label ='Outlet temperature of internal heat exchanger')
plt.xlabel('Time(Day|Hour|minute)',fontsize=20, color = 'r')
plt.ylabel("Temperature(℃)",fontsize =20, color = 'g')
 
plt.title('Inlet and Outlet temperature of internal heat exchanger',fontsize =20) 

plt.tight_layout()
plt.grid() 
plt.legend()


plt.show()

#Compressor

com_in = ass3['RT03']
com_out = ass3['RT07']

plt.figure(figsize=(12,8))
plt.plot(Time,com_in,label='Inlet temperature of compressor')
plt.plot(Time,com_out, label ='Outlet temperature of compressor')
plt.xlabel('Time(Day|Hour|minute)',fontsize=20, color = 'r')
plt.ylabel("Temperature(℃)",fontsize =20, color = 'g')
 
plt.title('Inlet and Outlet temperature of compressor',fontsize =20)

plt.tight_layout()
plt.grid() 
plt.legend()

#Gas cooler

gas_in = ass3['RT07']
gas_out = ass3['RT04']
gas_out1 = ass3['TT01']

plt.figure(figsize=(12,8))
plt.plot(Time,gas_in,label='Inlet temperature of gas cooler')
plt.plot(Time,gas_out, label ='Outlet temperature of gas cooler')
plt.plot(Time,gas_out1, label ='Outlet temperature of gas cooler t1')
plt.xlabel('Time(Day|Hour|minute)',fontsize=20, color = 'r')
plt.ylabel("Temperature(℃)",fontsize =20, color = 'g')
 
plt.title('Inlet and Outlet temperature of gas cooler',fontsize =20)

plt.tight_layout()
plt.grid() 
plt.legend()

#Internal heat exchanger high

ihe_in = ass3['RT04']
ihe_out = ass3['RT01']


plt.figure(figsize=(12,8))
plt.plot(Time,ihe_in,label='Inlet temperature of internal heat exchanger')
plt.plot(Time,ihe_out, label ='Outlet temperature of internal heat exchanger')

plt.xlabel('Time(Day|Hour|minute)',fontsize=20, color = 'r')
plt.ylabel("Temperature(℃)",fontsize =20, color = 'g')
 
plt.title('Inlet and Outlet temperature of high side of internal heat exchanger',fontsize =20)

plt.tight_layout()
plt.grid() 
plt.legend()

plt.show()


#Internal heat exchanger high difference 

high_in_diff = ass3['RT04']  - ass3['RT03'] 
high_out_diff = ass3['RT01'] - ass3['RT01']


plt.figure(figsize=(12,8))
plt.plot(Time,high_in_diff,label='high in to low out diff')
plt.plot(Time,high_out_diff, label ='high out to low in diff')

plt.xlabel('Time(Day|Hour|minute)',fontsize=20, color = 'r')
plt.ylabel("Temperature(℃)",fontsize =20, color = 'g')
 
plt.title('Difference in temperature for high and low side of internal heat exchanger',fontsize =20) 

plt.tight_layout()
plt.grid() 
plt.legend()

plt.show()


#Expansion valve

ex_in = ass3['RT01']
ex_out = ass3['RT02']


plt.figure(figsize=(12,8))
plt.plot(Time,ex_in,label='Expansion valve inlet')
plt.plot(Time,ex_out, label ='Expansion valve outlet')

plt.xlabel('Time(Day|Hour|minute)',fontsize=20, color = 'r')
plt.ylabel("Temperature(℃)",fontsize =20, color = 'g')
 
plt.title('Inlet and outlet temperature of expansion valve',fontsize =20)

plt.tight_layout()
plt.grid() 
plt.legend()

plt.show()


#water loop
wl_in = ass3['TT03']
wl_out = ass3['TT02']


plt.figure(figsize=(12,8))
plt.plot(Time,wl_in,label='Water loop inlet')
plt.plot(Time,wl_out, label ='Water loop outlet')

plt.xlabel('Time(Day|Hour|minute)',fontsize=20, color = 'r')
plt.ylabel("Temperature(℃)",fontsize =20, color = 'g')
plt.title('Inlet and Outlet temperature of water loop',fontsize =20)
 

plt.tight_layout()
plt.grid() 
plt.legend()

plt.show()


plt.figure(figsize=(12,8))
en_1 = (2659.2 * ex_in + 191325)/1000 
en_2 = (158902 * np.exp(0.0393 * ex_out))/1000
en_3 = (- 1038.8 * liq_in + 438652)/1000
en_4 = (438304 * np.exp(-0.002 * liq_out))/1000
en_5 = (848.98 * com_in + 439509)/1000
en_6 = (1103.05 * com_out + 400154)/1000
en_7 = (4530.2 * gas_out + 144225)/1000

plt.plot(Time,en_1, label='Enthalpy at inlet of expansion valve')
plt.plot(Time,en_2, label='Enthalpy at outlet of expansion valve')
plt.plot(Time,en_3, label ='Enthalpy at inlet of the liquid separator')
plt.plot(Time,en_4, label ='Enthalpy at outlet of the liquid separator')
plt.plot(Time,en_5, label ='Enthalpy at inlet of compressor')
plt.plot(Time,en_6, label ='Enthalpy at outlet of compressor')
plt.plot(Time,en_7, label='Enthalpy at outlet of gas cooler')

plt.xlabel('Time(Day|Hour|minute)',fontsize=20, color = 'r')
plt.ylabel("Enthaly (kJ/kg)",fontsize =20, color = 'g')
plt.title('Enthalpy at different measurement points',fontsize =20)
 

plt.tight_layout()
plt.grid() 
plt.legend(fontsize =16)

plt.show()



#heating rate 


flowheat = 300 / 3600
tempdi =  ass3['TT02'] -ass3['TT03']
c_p = 4.20

Q_h = flowheat * c_p * tempdi 




plt.figure(figsize=(12,8))
gas_e = ass3['GascoolerHeatrate']
plt.plot(Time,Q_h, label='Heat rate calculated toward water side')
plt.plot(Time,ass3['GascoolerHeatrate'],label= 'Heat rate measured of gas cooler')

plt.xlabel('Time(Day|Hour|minute)',fontsize=20, color = 'r')
plt.ylabel("Heat rate (kW)",fontsize =20, color = 'g')
plt.title('Heat rejection rate of gas cooler',fontsize =20)
 

plt.tight_layout()
plt.grid() 
plt.legend(fontsize =16)

plt.show()


#cop
plt.figure(figsize=(12,8))

comppower = ass3['CO2compressorpower']/1000

COP = Q_h  / comppower
plt.plot(Time,COP, label='COP of the compressor')


plt.xlabel('Time(Day|Hour|minute)',fontsize=20, color = 'r')
plt.ylabel("COP",fontsize =20, color = 'g')
plt.title('COP of the heat pump',fontsize =20)
 

plt.tight_layout()
plt.grid() 
plt.legend(fontsize =16)

plt.show()
#time requirements
 
cw = 4.2
rho = 1000
v = 0.001
tf = 60 
tb = 10

Q_h_1 = 3.2
#mode from 12:15, 13:00
print(Q_h_1)

t = (23.33 /Q_h_1)
print('The time required to heat the water tank from 10 to 60 (℃) is ')
print(t)



#lmtd

inlet_water =ass3['TT03'].mean()
print(inlet_water)
outlet_water = ass3['TT02'].mean()
print(outlet_water)
gc_in = ass3['RT07'].mean()
print(gc_in)
gc_out = ass3['RT04'].mean()
print(gc_out)

del2 = (gc_in + 273) - (outlet_water + 273)
print(del2)
del1 = (gc_out +273) - (inlet_water +273)
print(del1)

LMTD = del1 - del2/np.log(del1/del2)

UA = Q_h_1*1000 / LMTD 

print(UA)


plt.figure(figsize=(12,8))
inlet_water1 =ass3['TT03']
outlet_water1 = ass3['TT02']
gc_in1 = ass3['RT07']
gc_out1 = ass3['RT04']


del21 = (gc_in1 + 273) - (outlet_water1 + 273)

del11 = (gc_out1 +273) - (inlet_water1 +273)


LMTD1 = del11 - del21/np.log(del11/del21)

UA1 = Q_h_1*1000 / LMTD1 

plt.plot(Time,UA1, label='UA (w/k)')



plt.xlabel('Time(Day|Hour|minute)',fontsize=20, color = 'r')
plt.ylabel("UA (w/k)",fontsize =20, color = 'g')
plt.title('(UA (w/k) of the gas cooler',fontsize =20)
 

plt.tight_layout()
plt.grid() 
plt.legend(fontsize =16)
plt.show()


plt.figure(figsize=(12,8))
ret = ass3['TT01'] 
tr = ass3['ExpansionvalvePV']


fig, ax = plt.subplots(figsize = (12, 8))
plt.title('Throttling valve opening % vs COP',fontsize =20)
 

ax2 = ax.twinx()
ax.plot(Time, COP, color = 'g',label='COP')
ax2.plot(Time, tr, color = 'b',label='throttling valve opening %')
 

ax.set_xlabel('Time(Day|Hour|minute)', color = 'r',fontsize =20)
ax.set_ylabel('COP', color = 'g',fontsize =20)
 

ax2.set_ylabel('Expansion valve opening (%)', color = 'b',fontsize =20)
 
plt.tight_layout()
 
plt.grid()
plt.show()


fig, ax = plt.subplots(figsize = (12, 8))
plt.title('Return temperature (℃) vs COP',fontsize =20)
 

ax2 = ax.twinx()
ax.plot(Time, ret, color = 'g',label='return temperature')
ax2.plot(Time, COP, color = 'b',label='COP')
 

ax.set_xlabel('Time(Day|Hour|minute)', color = 'r',fontsize =20)
ax.set_ylabel('Return temperature (℃)', color = 'g',fontsize =20)
 

ax2.set_ylabel('COP', color = 'b',fontsize =20)
 
plt.tight_layout()
 
plt.grid()
plt.show()



fig, ax = plt.subplots(figsize = (12, 8))
plt.title('Return temperature (℃)vs heating rate (kW)',fontsize =20)
 

ax2 = ax.twinx()
ax.plot(Time, ret, color = 'g',label='return temperature')
ax2.plot(Time, Q_h, color = 'b',label='Heating rate')
 

ax.set_xlabel('Time(Day|Hour|minute)', color = 'r',fontsize =20)
ax.set_ylabel('Return temperature (℃)', color = 'g',fontsize =20)
 

ax2.set_ylabel('Heating rate (kW)', color = 'b',fontsize =20)
 
plt.tight_layout()
 
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize = (12, 8))
plt.title('Return temperature (℃)vs compressor power (kW)',fontsize =20)
 

ax2 = ax.twinx()
ax.plot(Time, ret, color = 'g')
ax2.plot(Time, comppower , color = 'b')
 

ax.set_xlabel('Time(Day|Hour|minute)', color = 'r',fontsize =20)
ax.set_ylabel('Return temperature (℃)', color = 'g',fontsize =20)
 

ax2.set_ylabel('Compressor power (kW)', color = 'b',fontsize =20)
 
plt.tight_layout()
 
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize = (12, 8))
plt.title('Expansion valve opening % vs compressor power (kW)',fontsize =20)
 

ax2 = ax.twinx()
ax.plot(Time, tr, color = 'g')
ax2.plot(Time, comppower , color = 'b')
 

ax.set_xlabel('Time(Day|Hour|minute)', color = 'r',fontsize =20)
ax.set_ylabel('Expansion valve opening %', color = 'g',fontsize =20)
 

ax2.set_ylabel('Compressor power (kW)', color = 'b',fontsize =20)
 
plt.tight_layout()
 
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize = (12, 8))
plt.title('Pressure (bar) vs Expansion valve opening % ',fontsize =20)
 

ax2 = ax.twinx()
ax.plot(Time,ass3['Highpressure'], color = 'g')
ax2.plot(Time, tr , color = 'b')
 

ax.set_xlabel('Time(Day|Hour|minute)', color = 'r',fontsize =20)
ax.set_ylabel('Pressure(bar)', color = 'g',fontsize =20)
ax2.set_ylabel('Expansion valve opening %', color = 'b',fontsize =20)
 
plt.tight_layout()
 
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize = (12, 8))
plt.title('Expansion valve opening % vs COP',fontsize =20)
 

ax2 = ax.twinx()
ax.plot(Time, tr, color = 'g')
ax2.plot(Time, COP , color = 'b')
 

ax.set_xlabel('Time(Day|Hour|minute)', color = 'r',fontsize =20)
ax.set_ylabel('Expansion valve opening %', color = 'g',fontsize =20)
 

ax2.set_ylabel('COP', color = 'b',fontsize =20)
 
plt.tight_layout()
 
plt.grid()
plt.show()


fig, ax = plt.subplots(figsize = (12, 8))
plt.title('Expansion valve opening % vs COP',fontsize =20)
 

ax2 = ax.twinx()
ax.plot(Time, tr, color = 'g')
ax2.plot(Time, COP , color = 'b')
 

ax.set_xlabel('Time(Day|Hour|minute)', color = 'r',fontsize =20)
ax.set_ylabel('Expansion valve opening %', color = 'g',fontsize =20)
 

ax2.set_ylabel('COP', color = 'b',fontsize =20)
 
plt.tight_layout()
 
plt.grid()
plt.show()
