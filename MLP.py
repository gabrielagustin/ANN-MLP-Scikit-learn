# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 10:16:04 2018

@author: gag 

"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
import statistics
import sklearn
import statsmodels.formula.api as smf

import selection
import lectura
import MLP_tuneParameters


   
def mlp_SAR_SMAP(porc, file, type, rand):
    print("Modelo MLP")
    if (type == "etapa1"):
        data = lectura.lecturaCompletaMLP_etapa1_SAR_SMAP(file)
        print(data)
        varCal = 'SM_CONAE'
        varVal = 'SM_CONAE'
        # ---------------------------------------------------------
#        ## PARAMETROS CALCULADOS POR PRUEBA Y ERROR
#        indexC1 = 2
#        indexC2 = 3
#        acti = 'logistic'
#        sol = 'lbfgs'
#        iter = 1000
#        random_s = 9
#        lr = 'adaptive'
#        lr = 'invscaling'
#        l_rate = 0.001
#        alpa = 0.1
#        momen = 0.75
        
        
#        ## PARAMETROS calculados por busqueda
#        indexC1 = 2
#        indexC2 = 2
#        acti = 'logistic'
#        sol = 'adam'
#        iter = 1000
#        random_s = 9
#        lr = 'adaptive'
#        lr = 'invscaling'
#        l_rate = 0.0505
#        alpa = 0.02575
#        momen = 0.54


#### utilizando solver "LBFGS"
#{'early_stopping': False, 'alpha': 0.1, 'warm_start': False, 'random_state': 9,
# 'power_t': 0.5, 'hidden_layer_sizes': (2, 3), 'activation': 'logistic', 
# 'max_iter': 5000, 'solver': 'lbfgs'}
#169
#RMSE Cal:3.548665941613726
#R^2 Cal: 0.6650365326863088
#Validacion
#55
#RMSE Val:4.206217630781608
#R^2 Val: 0.48883273779379244        
        
#### utilizando solver "ADAM"
#{'max_iter': 1000, 'random_state': 9, 'alpha': 0.025750000000000002, 
#'warm_start': False, 'power_t': 0.5, 'solver': 'adam', 
#'learning_rate_init': 0.025750000000000002, 'hidden_layer_sizes': (4, 6),
# 'activation': 'logistic', 'early_stopping': False}
#169
#RMSE Cal:3.882153747871466
#R^2 Cal: 0.5990273696112527
#Validacion
#55
#RMSE Val:4.590650437025272
#R^2 Val: 0.41562003458080976


###---------------parametros para el modelo utilizando datos SAR de SMAP-------


#### utilizando solver "SGD"
#        indexC1 = 7
#        indexC2 = 7
#        acti = 'relu'
#        sol = 'sgd'
#        iter = 1000
#        random_s = 9
#        lr = 'invscaling'
#        l_rate = 0.1
#        alpa = 0.07525
#        momen = 0.7        

#{'learning_rate': 'adaptive', 'validation_fraction': 0.25, 'alpha': 0.001, 
#'hidden_layer_sizes': (3, 7), 'warm_start': False, 'power_t': 0.5,
# 'momentum': 0.7000000000000001, 'activation': 'relu', 'max_iter': 500,
# 'solver': 'sgd', 'learning_rate_init': 0.001, 'random_state': 9, 'early_stopping': True}
#169
#RMSE Cal:3.9618219521931817
#R^2 Cal: 0.5830226185373626
#Validacion
#55
#RMSE Val:4.340297836943199
#R^2 Val: 0.4828850684470303


        indexC1 = 3
        indexC2 = 7
        acti = 'relu'
        sol = 'sgd'
        iter = 1000
        l_rate = 0.001
        alpa = 0.001
        momen = 0.7
        random_s = 9
        val_frac = 0.25
        lr = 'adaptive'




# ---------------------------------------------------------
        np.random.seed(rand)
        dataNew = selection.shuffle(data)
        dataNew = dataNew.reset_index(drop=True)
        nRow = len(dataNew.index)
        numTraining=int(round(nRow)*porc)
        print("Cantidad de elementos para el calculo de coeff: " + str(numTraining))
        numTest=int((nRow)-numTraining)
        print("Cantidad de elementos para prueba: " +str(numTest))


        dataTraining =  dataNew.ix[:numTraining, :]
        dataTraining = selection.shuffle(dataTraining)
        dataTraining = dataTraining.reset_index(drop=True)
        #print dataTraining

        dataTest = dataNew.ix[numTraining + 1:, :]

        OldRange = (np.max(dataTraining.Ts_SMAP)  - np.min(dataTraining.Ts_SMAP))
        NewRange = (1 + 1)
        dataTraining.Ts_SMAP = (((dataTraining.Ts_SMAP - np.min(dataTraining.Ts_SMAP)) * NewRange) / OldRange) -1

        OldRange = (np.max(dataTraining.GPM)  - np.min(dataTraining.GPM))
        NewRange = (1 + 1)
        dataTraining.GPM = (((dataTraining.GPM - np.min(dataTraining.GPM)) * NewRange) / OldRange) -1

        OldRange = (np.max(dataTraining.sigma0_vv_)  - np.min(dataTraining.sigma0_vv_))
        NewRange = (1 + 1)
        dataTraining.sigma0_vv_ = (((dataTraining.sigma0_vv_ - np.min(dataTraining.sigma0_vv_)) * NewRange) / OldRange) -1


        OldRange = (np.max(dataTest.Ts_SMAP)  - np.min(dataTest.Ts_SMAP))
        NewRange = (1 + 1)
        dataTest.Ts_SMAP = (((dataTest.Ts_SMAP - np.min(dataTest.Ts_SMAP)) * NewRange) / OldRange) -1

        OldRange = (np.max(dataTest.GPM)  - np.min(dataTest.GPM))
        NewRange = (1 + 1)
        dataTest.GPM = (((dataTest.GPM - np.min(dataTest.GPM)) * NewRange) / OldRange) -1

        OldRange = (np.max(dataTest.sigma0_vv_) - np.min(dataTest.sigma0_vv_))
        NewRange = (1 + 1)
        dataTest.sigma0_vv_ = (((dataTest.sigma0_vv_ - np.min(dataTest.sigma0_vv_)) * NewRange) / OldRange) -1


        yTraining = dataTraining[varCal]
        del dataTraining[varCal]
        xTraining = dataTraining

        yTest = dataTest['SM_CONAE']
        del dataTest[varCal]
        xTest = dataTest

            
#        print("------------------------------------------------------------------------")
#        print("------------------------------------------------------------------------")
#        print("------------------------------------------------------------------------")
#        print("-tune parameters")
#        
#        MLPmodel, yCalMLP, yAproxMLP = MLP_tuneParameters.tune_SAR_SMAP(xTraining,yTraining, xTest, yTest)
#        
#        
#        print("------------------------------------------------------------------------")
#        print("------------------------------------------------------------------------")
#        print("------------------------------------------------------------------------")
#        print("------------------------------------------------------------------------")
#        return

    print("indice capa 1: " + str(indexC1))
    print("indice capa 2: " + str(indexC2))
    
    reg = MLPRegressor(hidden_layer_sizes=(indexC1,indexC2), activation= acti, solver= sol, alpha=alpa,batch_size='auto',
           learning_rate=lr, learning_rate_init=l_rate, max_iter=iter, shuffle=True,
           random_state=random_s, tol=0.0001, verbose=False, warm_start=False, momentum=momen,
           nesterovs_momentum=True, early_stopping=True, validation_fraction=val_frac)




    reg = reg.fit(xTraining, yTraining)
    print("Estructura del MLP")
    for i in range (0,3):
        print(reg.coefs_[i])

    yCal = reg.predict(xTraining)
    print("AQUIIIII: " +str(yCal.shape))

    print ("------------------------------------------------------------------------")
    print("MLP Calibracion: ")
#    print("calibracion SMAP vs SMAP")
    rmse = statistics.RMSE(np.array(yTraining),np.array(yCal))
    print("RMSE:" + str(rmse))
    RR = sklearn.metrics.r2_score(yCal, yTraining)
    print("Coeficiente de Determinacion:" + str(RR))
    data2 = pd.DataFrame({'yCal' :yCal,'yTraining' :yTraining})
    RR = smf.ols('yCal ~ 1+ yTraining', data2).fit().rsquared
    print("R^2 222: "+str(RR))

    bias = statistics.bias(yTraining, yCal)
    print("Bias:" + str(bias))



    print("------------------------------------------------------------------------")
    print("MLP Validacion: ")
    yAprox = reg.predict(xTest)
    rmse = statistics.RMSE(np.array(yTest),np.array(yAprox))
    print("RMSE:" + str(rmse))
    RR = sklearn.metrics.r2_score(yTest, yAprox)
    #RR = smf.ols('yTest_SMAP ~ 1+ yAprox', data).fit().rsquared
    print("Coeficiente de Determinacion:" + str(RR))
    data2 = pd.DataFrame({'yTest_SMAP' :yTest,'yAprox' :yAprox})
    RR = smf.ols('yTest_SMAP ~ 1+ yAprox', data2).fit().rsquared
    print("R^2 222: "+str(RR))
    bias = statistics.bias(yTest, yAprox)
    print("Bias:" + str(bias))




    return reg, yCal, yAprox

    
