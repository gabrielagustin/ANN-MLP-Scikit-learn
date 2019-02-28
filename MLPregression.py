# -*- coding: utf-8 -*-
import lectura
import numpy as np
import selection
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
import statistics
import sklearn
import pandas as pd
import statsmodels.formula.api as smf
import MLPregresor_tuneParameters

#def mlp(fileCal, fileVal, type, rand):
def mlp(porc, file, type, rand):
    print("Modelo MLP")
    if (type == "etapa1"):
        data = lectura.lecturaCompletaMLP_etapa1(file)
        print(data)
        varCal = 'SM_CONAE'
        varVal = 'SM_CONAE'
        # ---------------------------------------------------------
        ## PARAMETROS CALCULADOS POR PRUEBA Y ERROR
        indexC1 = 3
        indexC2 = 4
        acti = 'relu'
        sol = 'sgd'
        iter = 1000
        random_s = 9
        lr = 'adaptive'
        l_rate = 0.01
        alpa = 0.015
        momen = 0.75

# ---------------------------------------------------------

        # PARAMETROS CALCULADOS POR Búsqueda aleatoria

        #####solver 'lbfgs'
        #indexC1 = 6
        #indexC2 = 4
        #acti = 'relu'
        #sol = 'lbfgs'
        #iter = 1000
        #alpa = 0.02575
        #random_s = 9
        ##### los siguientes lbfgs no los utiliza
        #lr = 'adaptive'
        #momen = 0.75
        #l_rate = 0.001

        # solver 'adams'
#        indexC1 = 4
#        indexC2 = 3
#        acti = 'relu'
#        sol = 'adam'
#        iter = 1000
#        random_s = 9
#        l_rate = 0.001
#        alpa = 0.001
#        # a los siguientes adam  no los utiliza
#        lr = 'adaptive'
#        momen = 0.75
#        val_frac = 0.1

        #### solver 'sgd'
        #indexC1 = 2
        #indexC2 = 4
        #acti = 'tanh'
        #sol = 'sgd'
        #iter = 1000
        #random_s = 9
        #l_rate = 0.078
        #alpa = 0.1
        #lr = 'invscaling'
        #momen = 0.63

        ### todos juntos
        #indexC1 = 5
        #indexC2 = 4
        #acti = 'relu'
        #sol = 'lbfgs'
        #iter = 1000
        #random_s = 9
        #l_rate = 0.078
        #alpa = 0.1
        #lr = 'adaptive'
        #momen = 0.36



        # ---------------------------------------------------------

        # PARAMETROS CALCULADOS POR BUSQUEDA EN CUADRICULA
        ####solver 'lbfgs'
        #indexC1 = 5
        #indexC2 = 3
        #acti = 'relu'
        #sol = 'lbfgs'
        #iter = 1000
        #alpa = 0.1
        #random_s = 9
        ##### los siguientes lbfgs no los utiliza
        #lr = 'adaptive'
        #momen = 0.75
        #l_rate = 0.001


        #### solver 'adams'
        #indexC1 = 6
        #indexC2 = 2
        #acti = 'relu'
        #sol = 'adam'
        #iter = 1000
        #random_s = 9
        #l_rate = 0.0257
        #alpa = 0.001
        ## a los siguientes adam  no los utiliza
        #lr = 'adaptive'
        #momen = 0.75

        #### solver 'sgd'
        #indexC1 = 4
        #indexC2 = 6
        #acti = 'relu'
        #sol = 'sgd'
        #iter = 1000
        #random_s = 9
        #l_rate = 0.02575
        #alpa = 0.1
        #lr = 'adaptive'
        #momen = 0.7

#### para el data set con resolucion de 1 km 
    if (type == "etapa2"):

        dataCal = lectura.lecturaCompletaMLP_etapa2(fileCal)
        dataVal = lectura.lecturaCompletaMLP_etapa2(fileVal)
        #print dataVal

        np.random.seed(0)
        dataNew = selection.shuffle(dataCal)
        dataTraining = dataNew.reset_index(drop=True)

        np.random.seed(0)
        dataNew = selection.shuffle(dataVal)
        dataTest = dataNew.reset_index(drop=True)

        varCal = 'SM_SMAP'
        #indexC1 = 8 ### lo obtengo con sigma0 a 5km con GPM y Et con bilinear, ndvi <0.8 pero con acti = 'logistic'
        #indexC2 = 0
        #### -----------------------------------------------------------------------
        #indexC1 = 5 ### lo obtengo con sigma0 a 5km con GPM con bilinear, ndvi <0.8
        #indexC2 = 5

        ### para HR
        #indexC1 = 5 ### lo obtengo con sigma0 a 5km con GPM y HR con bilinear, ndvi <0.8
        #indexC2 = 4
        #acti = 'relu'
        #sol = 'adam'
        #iter = 10000
        #l_rate = 0.0055
        #alpa = 0.0205
        #momen = 0.01
        #random_s = 5
        #lr = 'adaptive'

        #####-----------------------------------------------------------------


        ## para ET
        #indexC1 = 3
        #indexC2 = 6
        #acti = 'relu'
        #sol = 'adam'
        #iter = 1000
        #l_rate = 0.061
        #alpa = 0.0001
        #momen = 0.1
        #random_s = 1
        #lr = 'adaptive'


        indexC1 = 7
        indexC2 = 3
        acti = 'relu'
        sol = 'sgd'
        iter = 1000
        l_rate = 0.02575
        alpa = 0.0505
        momen = 0.7
        random_s = 9
        lr = 'adaptive'

        #{'warm_start': False, 'solver': 'sgd', 'hidden_layer_sizes': (7, 3), 'activation': 'relu',
         #'max_iter': 1000, 'power_t': 0.5, 'random_state': 9, 'early_stopping': True,
          #'alpha': 0.050500000000000003, 'momentum': 0.70000000000000007, 'learning_rate': 'adaptive', 'learning_rate_init': 0.025750000000000002}

        #{'warm_start': False, 'solver': 'sgd', 'learning_rate': 'adaptive', 'max_iter': 1000,
        #'power_t': 0.5, 'random_state': 9, 'learning_rate_init': 0.012, 'alpha': 0.023000000000000003,
         #'early_stopping': False, 'activation': 'relu', 'momentum': 0.81111111111111112, 'hidden_layer_sizes': (13, 9)}

    #{'warm_start': False, 'solver': 'sgd', 'learning_rate': 'adaptive', 'max_iter': 3000,
         #'power_t': 0.5, 'random_state': 9, 'learning_rate_init': 0.001, 'alpha': 0.012,
          #'early_stopping': False, 'activation': 'relu', 'momentum': 0.27777777777777779, 'hidden_layer_sizes': (2, 8)}

        #{'warm_start': False, 'hidden_layer_sizes': (14, 14), 'activation': 'relu',
         #'max_iter': 1000, 'power_t': 0.5, 'random_state': 9, 'early_stopping': False,
         #'alpha': 0.025750000000000002, 'solver': 'adam', 'learning_rate_init': 0.10000000000000001}

        #{'warm_start': False, 'hidden_layer_sizes': (8, 4), 'activation': 'relu',
        #'max_iter': 1000, 'power_t': 0.5, 'random_state': 9, 'early_stopping': False,
         #'alpha': 0.025750000000000002, 'solver': 'adam', 'learning_rate_init': 0.10000000000000001}


        # {'warm_start': False, 'hidden_layer_sizes': (11, 12), 'activation': 'relu',
        # 'max_iter': 1000, 'power_t': 0.5, 'random_state': 9, 'early_stopping': False,
        #'alpha': 0.001, 'solver': 'adam', 'learning_rate_init': 0.001}

        # {'warm_start': False, 'hidden_layer_sizes': (7, 8), 'activation': 'relu',
        #'max_iter': 1000, 'power_t': 0.5, 'random_state': 9, 'early_stopping': False,
        #'alpha': 0.025750000000000002, 'solver': 'adam', 'learning_rate_init': 0.001}



    if (type == "etapa3"):
            
        dataCal = lectura.lecturaCompletaMLP_etapa3(fileCal)
        dataVal = lectura.lecturaCompletaMLP_etapa3(fileVal)

#        print("MLP  #####################################################################")
#        print (dataCal.describe())
#        print("#####################################################################")



        
        np.random.seed(rand)
        dataNew = selection.shuffle(dataCal)
        dataTraining = dataNew.reset_index(drop=True)

        np.random.seed(rand)
        dataNew = selection.shuffle(dataVal)
        dataTest = dataNew.reset_index(drop=True)

        varCal = 'SM_SMAP'
        
#{'learning_rate': 'adaptive', 'power_t': 0.5, 'early_stopping': True, 'max_iter': 1000,
# 'alpha': 0.001, 'momentum': 0.7000000000000001, 'learning_rate_init': 0.001,
# 'activation': 'relu', 'warm_start': False, 'hidden_layer_sizes': (10, 7), 
# 'solver': 'sgd', 'random_state': 9}
#387
#RMSE Cal:6.113031748720126
#R^2 Cal: 0.5366600943086393
#Validacion
#258
#RMSE Val:8.589043218046823
#R^2 Val: 0.43175515438532297
#Presionar una tecla!!!!!


#        indexC1 = 10
#        indexC2 = 7
#        acti = 'relu'
#        sol = 'sgd'
#        iter = 1000
#        l_rate = 0.001
#        alpa = 0.001
#        momen = 0.7
#        random_s = 9
#        lr = 'adaptive'

#{'learning_rate': 'adaptive', 'activation': 'relu', 'solver': 'sgd',
# 'momentum': 0.8111111111111111, 'warm_start': False, 'learning_rate_init': 0.001,
# 'max_iter': 1000, 'validation_fraction': 0.25, 'power_t': 0.5, 
# 'early_stopping': True, 'hidden_layer_sizes': (7, 9), 'alpha': 0.012, 'random_state': 9}
#387
#RMSE Cal:6.0185914211555
#R^2 Cal: 0.5509886931750194
#Validacion
#258
#RMSE Val:8.28695689629607
#R^2 Val: 0.46981173811883914

#        indexC1 = 7
#        indexC2 = 9
#        acti = 'relu'
#        sol = 'sgd'
#        iter = 1000
#        l_rate = 0.001
#        alpa = 0.012
#        momen = 0.81
#        random_s = 9
#        val_frac = 0.25
#        lr = 'adaptive'


##### ------------------------- para datset con resolucion 9km-----------------
#{'early_stopping': True, 'validation_fraction': 0.25, 'power_t': 0.5, 
#'random_state': 9, 'hidden_layer_sizes': (11, 9), 'alpha': 0.1, 
#'learning_rate': 'adaptive', 'activation': 'logistic', 'max_iter': 1000,
# 'momentum': 0.7000000000000001, 'warm_start': False, 'learning_rate_init': 0.0505,
# 'solver': 'sgd'}
#387
#RMSE Cal:5.994875668274481
#R^2 Cal: 0.5541947269096251
#Validacion
#258
#RMSE Val:6.86925164817972
#R^2 Val: 0.6594478869081521


#        indexC1 = 11
#        indexC2 = 9
#        acti = 'logistic'
#        sol = 'sgd'
#        iter = 1000
#        l_rate = 0.0505
#        alpa = 0.1
#        momen = 0.7
#        random_s = 9
#        val_frac = 0.25
#        lr = 'adaptive'

##### ------------------------- para datset con resolucion 1km-----------------
#{'early_stopping': True, 'warm_start': False, 'learning_rate_init': 0.025750000000000002,
# 'power_t': 0.5, 'random_state': 9, 'learning_rate': 'adaptive', 'activation': 'relu',
# 'alpha': 0.07525000000000001, 'hidden_layer_sizes': (6, 9), 'validation_fraction': 0.25, 
# 'max_iter': 1000, 'momentum': 0.5, 'solver': 'sgd'}
#39637
#RMSE Cal:6.212011514535369
#R^2 Cal: 0.5493830155926851
#Validacion
#25668
#RMSE Val:6.965492355400995
#R^2 Val: 0.6510697858177239
#Presionar una tecla!!!!!


        indexC1 = 6
        indexC2 = 9
        acti = 'relu'
        sol = 'sgd'
        iter = 5000
        l_rate = 0.02575
        alpa = 0.07525
        momen = 0.5
        random_s = 9
        val_frac = 0.25
        lr = 'adaptive'

    if (type == "etapa1"):

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

        OldRange = (np.max(dataTraining.T_aire)  - np.min(dataTraining.T_aire))
        NewRange = (1 + 1)
        dataTraining.T_aire = (((dataTraining.T_aire - np.min(dataTraining.T_aire)) * NewRange) / OldRange) -1

        OldRange = (np.max(dataTraining.HR)  - np.min(dataTraining.HR))
        NewRange = (1 + 1)
        dataTraining.HR = (((dataTraining.HR - np.min(dataTraining.HR)) * NewRange) / OldRange) -1

        OldRange = (np.max(dataTraining.PP)  - np.min(dataTraining.PP))
        NewRange = (1 + 1)
        dataTraining.PP = (((dataTraining.PP - np.min(dataTraining.PP)) * NewRange) / OldRange) -1

        OldRange = (np.max(dataTraining.Sigma0)  - np.min(dataTraining.Sigma0))
        NewRange = (1 + 1)
        dataTraining.Sigma0 = (((dataTraining.Sigma0 - np.min(dataTraining.Sigma0)) * NewRange) / OldRange) -1


        OldRange = (np.max(dataTest.T_aire)  - np.min(dataTest.T_aire))
        NewRange = (1 + 1)
        dataTest.T_aire = (((dataTest.T_aire - np.min(dataTest.T_aire)) * NewRange) / OldRange) -1

        OldRange = (np.max(dataTest.HR)  - np.min(dataTest.HR))
        NewRange = (1 + 1)
        dataTest.HR = (((dataTest.HR - np.min(dataTest.HR)) * NewRange) / OldRange) -1

        OldRange = (np.max(dataTest.PP)  - np.min(dataTest.PP))
        NewRange = (1 + 1)
        dataTest.PP = (((dataTest.PP - np.min(dataTest.PP)) * NewRange) / OldRange) -1

        OldRange = (np.max(dataTest.Sigma0)  - np.min(dataTest.Sigma0))
        NewRange = (1 + 1)
        dataTest.Sigma0 = (((dataTest.Sigma0 - np.min(dataTest.Sigma0)) * NewRange) / OldRange) -1

        yTraining = dataTraining[varCal]
        del dataTraining[varCal]
        xTraining = dataTraining

        yTest_SMAP = dataTest['SM_CONAE']
        del dataTest[varCal]
        test_x = dataTest
        
        
        
#        print("------------------------------------------------------------------------")
#        print("------------------------------------------------------------------------")
#        print("------------------------------------------------------------------------")
#        print("-tune parameters")
#        
#        MLPmodel, yCalMLP, yAproxMLP = MLPregresor_tuneParameters.tune_SAR_SMAP(xTraining,yTraining, test_x, yTest_SMAP)
#        
#        
#        print("------------------------------------------------------------------------")
#        print("------------------------------------------------------------------------")
#        print("------------------------------------------------------------------------")
#        print("------------------------------------------------------------------------")
#    return        
        
        
        

    if (type == "etapa2" or type =="etapa3"):

#        print(dataTraining)


        OldRange = (np.max(dataTraining.T_s)  - np.min(dataTraining.T_s))
        NewRange = (1 + 1)
        dataTraining.T_s = (((dataTraining.T_s - np.min(dataTraining.T_s)) * NewRange) / OldRange) -1

        OldRange = (np.max(dataTraining.Et)  - np.min(dataTraining.Et))
        NewRange = (1 + 1)
        dataTraining.Et = (((dataTraining.Et - np.min(dataTraining.Et)) * NewRange) / OldRange) -1

        #OldRange = (np.max(dataTraining.HR)  - np.min(dataTraining.HR))
        #NewRange = (1 + 1)
        #dataTraining.HR = (((dataTraining.HR - np.min(dataTraining.HR)) * NewRange) / OldRange) -1

        OldRange = (np.max(dataTraining.PP)  - np.min(dataTraining.PP))
        NewRange = (1 + 1)
        dataTraining.PP = (((dataTraining.PP - np.min(dataTraining.PP)) * NewRange) / OldRange) -1

        OldRange = (np.max(dataTraining.Sigma0)  - np.min(dataTraining.Sigma0))
        NewRange = (1 + 1)
        dataTraining.Sigma0 = (((dataTraining.Sigma0 - np.min(dataTraining.Sigma0)) * NewRange) / OldRange) -1


        OldRange = (np.max(dataTest.T_s)  - np.min(dataTest.T_s))
        NewRange = (1 + 1)
        dataTest.T_s = (((dataTest.T_s - np.min(dataTest.T_s)) * NewRange) / OldRange) -1

        OldRange = (np.max(dataTest.Et)  - np.min(dataTest.Et))
        NewRange = (1 + 1)
        dataTest.Et = (((dataTest.Et - np.min(dataTest.Et)) * NewRange) / OldRange) -1

        #OldRange = (np.max(dataTest.HR)  - np.min(dataTest.HR))
        #NewRange = (1 + 1)
        #dataTest.HR = (((dataTest.HR - np.min(dataTest.HR)) * NewRange) / OldRange) -1


        OldRange = (np.max(dataTest.PP)  - np.min(dataTest.PP))
        NewRange = (1 + 1)
        dataTest.PP = (((dataTest.PP - np.min(dataTest.PP)) * NewRange) / OldRange) -1

        OldRange = (np.max(dataTest.Sigma0)  - np.min(dataTest.Sigma0))
        NewRange = (1 + 1)
        dataTest.Sigma0 = (((dataTest.Sigma0 - np.min(dataTest.Sigma0)) * NewRange) / OldRange) -1

        yTraining = dataTraining[varCal]
        del dataTraining[varCal]
        xTraining = dataTraining
        #print "datos de calibracion"

        yTest_SMAP = dataTest['SM_SMAP']
        del dataTest[varCal]
        test_x = dataTest
        #print "datos de validacion"
        #print test_x
        
        print("-----------------------------------------------")
        print(list(xTraining))
        print("-----------------------------------------------")


    #print "---------------------------------------------------------------------"

    print("indice capa 1: " + str(indexC1))
    print("indice capa 2: " + str(indexC2))
    if (type == "etapa1"):
        reg = MLPRegressor(hidden_layer_sizes=(indexC1,indexC2), activation= acti, solver= sol, alpha=alpa,batch_size='auto',
                   learning_rate=lr, learning_rate_init=l_rate, power_t=0.5, max_iter=iter, shuffle=True,
                   random_state=random_s, tol=0.0001, verbose=False, warm_start=False, momentum=momen,
                   nesterovs_momentum=True, early_stopping=False, beta_1=0.9, beta_2=0.999,
                   epsilon=1e-08)
    else:
        reg = MLPRegressor(hidden_layer_sizes=(indexC1,indexC2), activation= acti, solver= sol, alpha=alpa,batch_size='auto',
               learning_rate=lr, learning_rate_init=l_rate, max_iter=iter, shuffle=True,
               random_state=random_s, tol=0.0001, verbose=False, warm_start=False, momentum=momen,
               nesterovs_momentum=True, early_stopping=True, validation_fraction=val_frac)
           
    ### Aqui deberia implementar medotodo de validacion k-fold
#    print("validacion cruzada")
#    aaa = cross_val_score(reg, xTraining, yTraining, cv=5)
#    print(aaa)



    reg = reg.fit(xTraining, yTraining)
    print("Estructura del MLP")
    for i in range (0,3):
        print(reg.coefs_[i])

    yCal = reg.predict(xTraining)
    print("AQUIIIII: " +str(yCal.shape))

    print ("------------------------------------------------------------------------")
    print("MLP Calibracion: ")
    print("calibracion SMAP vs SMAP")
    rmse = statistics.RMSE(np.array(yTraining),np.array(yCal))
    print("RMSE:" + str(rmse))
    RR = sklearn.metrics.r2_score(yCal, yTraining)
    print("Coeficiente de Determinacion:" + str(RR))
    data2 = pd.DataFrame({'yCal' :yCal,'yTraining' :yTraining})
    RR = smf.ols('yCal ~ 1+ yTraining', data2).fit().rsquared
    print("R^2 222: "+str(RR))



    bias = statistics.bias(yTraining, yCal)
    print("Bias:" + str(bias))
    #print "calibracion SMAP vs CONAE"
    #rmse = statistics.RMSE(np.array(yTraining_CONAE),np.array(yCal))
    #print "RMSE:" + str(rmse)
    #RR = sklearn.metrics.r2_score(yTraining_CONAE, yCal)
    #print "Coeficiente de Determinacion:" + str(RR)
    #bias = statistics.bias(yTraining_CONAE, yCal)
    #print "Bias:" + str(bias)

    yAprox = reg.predict(test_x)

    print("------------------------------------------------------------------------")
    print("MLP Validacion: ")
    print("calibracion con SMAP/ validacion con SMAP")
    rmse = statistics.RMSE(np.array(yTest_SMAP),np.array(yAprox))
    print("RMSE:" + str(rmse))
    RR = sklearn.metrics.r2_score(yTest_SMAP, yAprox)
    #RR = smf.ols('yTest_SMAP ~ 1+ yAprox', data).fit().rsquared
    print("Coeficiente de Determinacion:" + str(RR))
    data2 = pd.DataFrame({'yTest_SMAP' :yTest_SMAP,'yAprox' :yAprox})
    RR = smf.ols('yTest_SMAP ~ 1+ yAprox', data2).fit().rsquared
    print("R^2 222: "+str(RR))
    bias = statistics.bias(yTest_SMAP, yAprox)
    print("Bias:" + str(bias))
#
#
#    print("pearson")
#    print(np.corrcoef(yAprox,yTest_SMAP)[1,0])


    #v1 = yTest
    #v2 = yAprox

    #z = np.polyfit(v1,v2, 1)
    #g = np.poly1d(z)
    #cor = np.corrcoef(v1,v2)[0,1]
    #if (cor >0 ):
        #cor=(cor)*(cor)
    #else:
        #cor=(cor*(-1))*(cor*(-1))

    #fig = plt.figure(1,facecolor="white")
    #ax1 = fig.add_subplot(111,aspect='equal')
    #ax1.plot(v1,g(v1),'black')
    #ax1.text(10, 30, 'R^2=%5.3f' % RR, fontsize=12)
    #ax1.text(10, 28, 'r^2=%5.3f' % cor, fontsize=12)
    #ax1.set_xlabel("observed value [% GSM]",fontsize=12)
    #ax1.set_ylabel("estimated value [% GSM]",fontsize=12)
    ##ax1.set_xlabel("valor observado [% GSM]",fontsize=12)
    ##ax1.set_ylabel("valor estimado [% GSM]",fontsize=12)
    ##ax1.scatter(x, y, s=10, c='b', marker="s", label='real')
    #ax1.scatter(yTest,yAprox, s=10,color='black',linewidth=3)# c='r', marker="o", label='NN Prediction')
    #ax1.axis([5,45, 5,45])
    #plt.grid(True)

    #xx = np.linspace(0,len(yTest),len(yTest))


    #fig = plt.figure(2,facecolor="white")
    #fig0 = fig.add_subplot(111)
    #fig0.text(0, 15, 'RMSE=%5.3f' % rmse, fontsize=12)


    #fig0.scatter(xx, yTest, color='blue',linewidth=3,label='SM')
    #fig0.scatter(xx, yAprox,s=65, color='black',marker = "*",label='SM_Aprox')
    #fig0.legend(loc=1, fontsize = 'medium')
    ##fig0.set_xlabel("Samples",fontsize=12)
    ##fig0.set_ylabel("Soil moisture [% GSM]",fontsize=12)
    #fig0.set_xlabel("muestras",fontsize=12)
    #fig0.set_ylabel("humedad de suelo [% GSM]",fontsize=12)
    #plt.show()

    return reg, yCal, yAprox
    
    
####---------------------------------------------------------------------------
    
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
##        lr = 'invscaling'
#        l_rate = 0.001
#        alpa = 0.1
#        momen = 0.75
        
        
        ## PARAMETROS calculados por busqueda
#        indexC1 = 2
#        indexC2 = 2
#        acti = 'logistic'
#        sol = 'adam'
#        iter = 1000
#        random_s = 9
#        lr = 'adaptive'
##        lr = 'invscaling'
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
#        MLPmodel, yCalMLP, yAproxMLP = MLPregresor_tuneParameters.tune_SAR_SMAP(xTraining,yTraining, xTest, yTest)
#        
#        
#        print("------------------------------------------------------------------------")
#        print("------------------------------------------------------------------------")
#        print("------------------------------------------------------------------------")
#        print("------------------------------------------------------------------------")
#    return

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

    
