# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Wed Oct 24 10:16:04 2018
@author: gag 
"""



import os
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import selection
import statistics


from sklearn.preprocessing import normalize
from sklearn import preprocessing

from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
import seaborn as sns


import functools
def conjunction(*conditions):
    return functools.reduce(np.logical_and, conditions)


def func(x, a, b,):
    return a*np.exp(-b*x)

###----------------------------
#### convert a range to another range
#OldRange = (OldMax - OldMin)  
#NewRange = (NewMax - NewMin)  
#NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
###----------------------------


def normalizado(c):
    min = np.min(c)
    max = np.max(c)
    new = (c -min)/(max-min)
    OldRange = (max  - min)
    NewRange = (1 - 0.1)
    new = (((c - min) * NewRange) / OldRange) + 0.1
    return new
    


def lecturaCompleta_etapa1_SAR_SMAP_simple(file):
    data = pd.read_csv(file, sep=',', decimal=",")
    print ("Numero inicial de muestras: " + str(len(data)))
    del data['Date']
    print("aquiiii")
    del data['wkt_geom']
    del data['ID_DISPOSI']
    data['SM_CONAE'] = data['SM_CONAE']*100
    data['SM_SMAP'] = data['SM_SMAP']*100
    data['Ts_SMAP'] = data['Ts_SMAP']-273.15
#    data['$\sigma^0_{hh}$'] = 10*np.log10(data['$\sigma^0_{hh}$'])
#    data['$\sigma^0_{vv}$'] = 10*np.log10(data['$\sigma^0_{vv}$'])
    data.GPM = data.GPM * 0.1
#    del data['GPM']
    del data['Ts_CONAE']
    del data['SM_SMAP']

    print(data.describe())
    print("This is how you pause")
    input()


    #graph3(data)
    #data.to_csv('/home/gag/Desktop/salida.csv')

    print ("maximo humedad de suelo: " + str(np.max(data.SM_CONAE)))
    return data


####----------------------------------------------------------------------------
def lecturaCompleta_etapa1_SAR_SMAP(file):
    #### para el archivo csv cuando quiero obtener las salidas de la matriz
    #### de correlacion $\sigma^0_{hh}$

    data = pd.read_csv(file, sep=',', decimal=",")
    print ("Numero inicial de muestras: " + str(len(data)))
    del data['Date']
    print("aquiiii")
    del data['wkt_geom']
    del data['ID_DISPOSI']
    data['SM_CONAE'] = data['SM_CONAE']*100
    data['SM_SMAP'] = data['SM_SMAP']*100
    data['Ts_SMAP'] = data['Ts_SMAP']-273.15
    data['sigma0_hh_'] = 10*np.log10(data['sigma0_hh_'])
    data['sigma0_vv_'] = 10*np.log10(data['sigma0_vv_'])
    data.GPM = data.GPM * 0.1
    ###------------------------------
#    del data['Sigma0_30m']
    ###----------------------------
    #del data['Sigma0']
    #data = data.rename(index=str, columns={"Sigma0_30m":"Sigma0"})
    #print data
    # se filtra el rango de valores Humedad de suelo de conae
#    perc10SM = math.ceil(np.percentile(data.SM_CONAE, 0))
#    print ("percentile humedad 5: " + str(perc10SM))
#    perc90SM = math.ceil(np.percentile(data.SM_CONAE, 95))
#    print ("percentile humedad 90: " + str(perc90SM))
#    print ("Filtro por humedad")
#    dataNew = data[(data.SM_CONAE >= perc10SM) & (data.SM_CONAE <= 44)]
#    data = dataNew
#
#
#    print ("Numero de muestras: " + str(len(data)))


#    ## se filtra el rango de valores de backscattering
#    perc5Back = math.ceil(np.percentile(data.Sigma0,0))
#    print ("percentile back 5: " + str(perc5Back))
#    perc90Back = math.ceil(np.percentile(data.Sigma0, 95))
#    print ("percentile back 95: " + str(perc90Back))
#    dataNew = data[(data.Sigma0 > perc5Back) & (data.Sigma0 < perc90Back)]
#    data = dataNew
#    print ("Numero de muestras: " + str(len(data)))
#
#
#    # se filtra el rango de valores de HR
#    perc5HR = math.ceil(np.percentile(data.HR,0))
#    print ("percentile HR 5: " + str(perc5HR))
#    perc90HR = math.ceil(np.percentile(data.HR, 95))
#    print ("percentile HR 95: " + str(perc90HR))
#    dataNew = data[(data.HR > perc5HR) & (data.HR < perc90HR)]
#    data = dataNew
#
#    print ("Numero de muestras: " + str(len(data)))
#
#    ## se filtra el rango de valores de T_aire
#    perc5Ta = math.ceil(np.percentile(data.T_aire,0))
#    print ("percentile Ta 5: " + str(perc5Ta))
#    perc90Ta = math.ceil(np.percentile(data.T_aire, 95))
#    print ("percentile Ta 95: " + str(perc90Ta))
#    dataNew = data[(data.T_aire > perc5Ta) & (data.T_aire < perc90Ta)]
#    data = dataNew
#
#    print ("Numero de muestras: " + str(len(data)))
#
#
#    ### se filtra el rango de valores de Tension_va
#    perc5Tv = math.ceil(np.percentile(data.Tension_va,0))
#    print ("percentile Tv 5: " + str(perc5Tv))
#    perc90Tv = math.ceil(np.percentile(data.Tension_va, 95))
#    print ("percentile Tv 95: " + str(perc90Tv))
#    dataNew = data[(data.Tension_va > perc5Tv) & (data.Tension_va < perc90Tv)]
#    data = dataNew
#
#    print ("Numero de muestras: " + str(len(data)))
#
#    ### se filtra el rango de valores de RSOILTEMPC
#    perc5Ts = math.ceil(np.percentile(data.RSOILTEMPC,0))
#    print ("percentile Ts 5: " + str(perc5Ts))
#    perc90Ts = math.ceil(np.percentile(data.RSOILTEMPC, 97))
#    print ("percentile Ts 95: " + str(perc90Ts))
#    dataNew = data[(data.RSOILTEMPC > perc5Ts) & (data.RSOILTEMPC < perc90Ts)]
#    data = dataNew
#    print ("Numero de muestras: " + str(len(data)))
#
#
#    print ("max PP: " + str(np.max(data.PP)))
#    print ("min PP: " + str(np.min(data.PP)))
#
#
#    del data['FECHA_HORA']
#    del data['ID_DISPOSI']
    print(data.describe())
#    input()
    ### se normalizan entre 0 y 1 las variables 
    data.GPM = normalizado(data.GPM)
    data.sigma0_vv_ = normalizado(data.sigma0_vv_)
    data.sigma0_hh_ = normalizado(data.sigma0_hh_)

    data.Ts_CONAE = normalizado(data.Ts_CONAE)
    data.Ts_SMAP = normalizado(data.Ts_SMAP)
    

    print("Estadisticas estandarizados entre 0 y 1: ")
    print(data.describe())
    ## se filtran los dispositivos considerados que no se encuentran operativos
    ### por la gente de conae 122 124 128
    #dataNew = data[(data.ID_DISPOSI != 122)]
    #data = dataNew
    #dataNew = data[(data.ID_DISPOSI != 124)]
    #data = dataNew
    #dataNew = data[(data.ID_DISPOSI != 128)]
    #data = dataNew
    #print "Filtro por estaciones malas"
    #print "Numero de muestras: " + str(len(data))
#    print('ACAAAAAAAAAAAAAAAAAAAAAAAAAAA')
#    print(data.Sigma0)
#    print("This is how you pause")
#    input()



    ###-----------------------------------------------------------------------
    ### reglas para filtrar los datos
    ## se filtra el rango de valores de NDVI
    #dataNew = data[ (data.NDVI_30m_B > 0.0) & (data.NDVI_30m_B < 0.50)]

#    dataNew = data[ (data.NDVI > 0.0) & (data.NDVI < 0.51)]
#    data = dataNew
#    print ("Filtro por NDVI")
#    print ("Numero de muestras: " + str(len(data)))
#
#    print("Estadisticas estandarizados entre 0 y 1: ")
#    print(data.describe())
#    #del data['SM10Km_PCA']
#    #del data['SMAP']
#    del data['NDVI']
#
#
#
#    #statistics(data)
#    #statistics(data)
#    ### se aplica el logaritmo a las variables 
    data.SM_CONAE = np.log10(data.SM_CONAE)
    data.Ts_CONAE = np.log10(data.Ts_CONAE)
    data.Ts_SMAP = np.log10(data.Ts_SMAP)    
    
    
#
#    del data['RSOILTEMPC']
#    del data['Tension_va']
    del data['SM_SMAP']
    del data['Ts_CONAE']
#    del data['Ts_SMAP']
    del data['sigma0_hh_']
#    del data['sigma0_vv_']

    print(data.describe())
    print("This is how you pause")
    input()


    #graph3(data)
    #data.to_csv('/home/gag/Desktop/salida.csv')

    print ("maximo humedad de suelo: " + str(np.max(data.SM_CONAE)))
    return data


