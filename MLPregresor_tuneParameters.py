
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 10:16:04 2018
@author: gag 

"""



import numpy as np
import lectura
import selection
import statistics
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import itertools
import pandas as pd
import statsmodels.formula.api as smf


from scipy import stats
from sklearn.grid_search import RandomizedSearchCV




#scaler = StandardScaler()

#file = "tabla_calibration_validation.csv"
#data = lectura.lecturaCompletaMLP_etapa1(file)


def tune(fileCal, fileVal):

    #file = 'tabla_Completa.csv'

    #fileCal = 'tabla_completa_Calibracion.csv'
    #fileVal = 'tabla_completa_Validacion.csv'

    #dataCal = lectura.lecturaCompleta_etapa2(fileCal)
    #dataVal = lectura.lecturaCompleta_etapa2(fileVal)
#    dataCal = lectura.lecturaCompletaMLP_etapa2(fileCal)
#    dataVal = lectura.lecturaCompletaMLP_etapa2(fileVal)


    dataCal = lectura.lecturaCompletaMLP_etapa3(fileCal)
    dataVal = lectura.lecturaCompletaMLP_etapa3(fileVal)

#    print(dataVal)

    np.random.seed(0)
    dataNew = selection.shuffle(dataCal)
    dataNew = dataNew.reset_index(drop=True)
    dataCal =dataNew

    #varCal = 'SM_CONAE'
    varCal = 'SM_SMAP'
    y_train = dataCal[varCal]
    del dataCal[varCal]
    X_train = dataCal


    np.random.seed(0)
    dataNew = selection.shuffle(dataVal)
    dataNew = dataNew.reset_index(drop=True)
    dataVal =dataNew

    #varCal = 'SM_CONAE'
    varCal = 'SM_SMAP'
    y_test = dataVal[varCal]
    del dataVal[varCal]
    X_test = dataVal

#    print (y_train)




    #X_train, X_test, y_train, y_test = train_test_split(
        #X, y, random_state=0) # por defecto es 75-25

    #print len(y_train)
    #print len(y_test)



    #OldRange = (np.max(X_train.T_aire)  - np.min(X_train.T_aire))
    #NewRange = (1 + 1)
    #X_train.T_aire = (((X_train.T_aire - np.min(X_train.T_aire)) * NewRange) / OldRange) -1

    #OldRange = (np.max(X_train.HR)  - np.min(X_train.HR))
    #NewRange = (1 + 1)
    #X_train.HR = (((X_train.HR - np.min(X_train.HR)) * NewRange) / OldRange) -1

    #OldRange = (np.max(X_train.PP)  - np.min(X_train.PP))
    #NewRange = (1 + 1)
    #X_train.PP = (((X_train.PP - np.min(X_train.PP)) * NewRange) / OldRange) -1

    #OldRange = (np.max(X_train.Sigma0)  - np.min(X_train.Sigma0))
    #NewRange = (1 + 1)
    #X_train.Sigma0 = (((X_train.Sigma0 - np.min(X_train.Sigma0)) * NewRange) / OldRange) -1


    #OldRange = (np.max(X_test.T_aire)  - np.min(X_test.T_aire))
    #NewRange = (1 + 1)
    #X_test.T_aire = (((X_test.T_aire - np.min(X_test.T_aire)) * NewRange) / OldRange) -1

    #OldRange = (np.max(X_test.HR)  - np.min(X_test.HR))
    #NewRange = (1 + 1)
    #X_test.HR = (((X_test.HR - np.min(X_test.HR)) * NewRange) / OldRange) -1

    #OldRange = (np.max(X_test.PP)  - np.min(X_test.PP))
    #NewRange = (1 + 1)
    #X_test.PP = (((X_test.PP - np.min(X_test.PP)) * NewRange) / OldRange) -1

    #OldRange = (np.max(X_test.Sigma0)  - np.min(X_test.Sigma0))
    #NewRange = (1 + 1)
    #X_test.Sigma0 = (((X_test.Sigma0 - np.min(X_test.Sigma0)) * NewRange) / OldRange) -1





    OldRange = (np.max(X_train.T_s)  - np.min(X_train.T_s))
    NewRange = (1 + 1)
    X_train.T_s = (((X_train.T_s - np.min(X_train.T_s)) * NewRange) / OldRange) -1

    OldRange = (np.max(X_train.Et)  - np.min(X_train.Et))
    NewRange = (1 + 1)
    X_train.Et = (((X_train.Et - np.min(X_train.Et)) * NewRange) / OldRange) -1

    #OldRange = (np.max(X_train.HR)  - np.min(X_train.HR))
    #NewRange = (1 + 1)
    #X_train.HR = (((X_train.HR - np.min(X_train.HR)) * NewRange) / OldRange) -1

    OldRange = (np.max(X_train.PP)  - np.min(X_train.PP))
    NewRange = (1 + 1)
    X_train.PP = (((X_train.PP - np.min(X_train.PP)) * NewRange) / OldRange) -1

    OldRange = (np.max(X_train.Sigma0)  - np.min(X_train.Sigma0))
    NewRange = (1 + 1)
    X_train.Sigma0 = (((X_train.Sigma0 - np.min(X_train.Sigma0)) * NewRange) / OldRange) -1


    OldRange = (np.max(X_test.T_s)  - np.min(X_test.T_s))
    NewRange = (1 + 1)
    X_test.T_s = (((X_test.T_s - np.min(X_test.T_s)) * NewRange) / OldRange) -1

    #OldRange = (np.max(X_test.HR)  - np.min(X_test.HR))
    #NewRange = (1 + 1)
    #X_test.HR = (((X_test.HR - np.min(X_test.HR)) * NewRange) / OldRange) -1

    OldRange = (np.max(X_test.Et)  - np.min(X_test.Et))
    NewRange = (1 + 1)
    X_test.Et = (((X_test.Et - np.min(X_test.Et)) * NewRange) / OldRange) -1

    OldRange = (np.max(X_test.PP)  - np.min(X_test.PP))
    NewRange = (1 + 1)
    X_test.PP = (((X_test.PP - np.min(X_test.PP)) * NewRange) / OldRange) -1

    OldRange = (np.max(X_test.Sigma0)  - np.min(X_test.Sigma0))
    NewRange = (1 + 1)
    X_test.Sigma0 = (((X_test.Sigma0 - np.min(X_test.Sigma0)) * NewRange) / OldRange) -1




    mlp = MLPRegressor()
    ### solver 'lbfgs'
#    param_grid = {'hidden_layer_sizes': [i for i in itertools.product(range(1,7),repeat=2)],
#                  #'activation': ['relu', 'tanh', 'logistic'],
#                  'activation': ['relu'],
#                  'solver': ['lbfgs'],
#                  'power_t': [0.5],
#                  'alpha': np.linspace(0.001, 0.1, num=5),
#                  'max_iter': [5000],
#                  'early_stopping': [False],
#                  'warm_start': [False],
#                  'random_state': [9]
#                  }
    ### solver 'adam'
#    param_grid = {'hidden_layer_sizes': [i for i in itertools.product(range(4,12),repeat=2)],
##                  'activation': ['relu', 'tanh', 'logistic'],
#                  'activation': ['logistic'],
#                  'solver': ['adam'],
#                  'learning_rate_init':np.linspace(0.001, 0.1, num=5),
#                  'power_t': [0.5],
#                  'alpha': np.linspace(0.001, 0.1, num=5),
#                  'max_iter': [50],
#                  'early_stopping': [True],
#                  'validation_fraction': [0.25],
#                  'warm_start': [False],
#                  'random_state': [9]
#                  }

    ### solver 'sgd'
#    param_grid = {'hidden_layer_sizes': [i for i in itertools.product(range(4,12),repeat=2)],
##                  'activation': ['relu', 'tanh', 'logistic'],
#                  'activation': ['relu'],
##                  'activation': ['logistic'],
#                  'solver': ['sgd'],
##                  'learning_rate': ['constant'],##, 'invscaling', 'adaptive'],
#                  'learning_rate': ['constant'],
#                  'learning_rate_init':np.linspace(0.001, 0.1, num=10),
#                  'power_t': [0.5],
#                  'alpha': np.linspace(0.001, 0.1, num=10),
#                  'momentum': np.linspace(0.1, 0.9, num=10), # cuando se usa 'sgd'
#                  'max_iter': [1000],
#                  'early_stopping': [True],
#                  'validation_fraction': [0.25],
#                  'warm_start': [False],
#                  'random_state': [9]
#                  }
#    Scoring='neg_mean_squared_error'
#    ### se utilza 3-fold cross validation
#    _GS = GridSearchCV(mlp, param_grid=param_grid, scoring=Scoring,
#                       cv=2, verbose=True, pre_dispatch=None)
#                      #, pre_dispatch='2*n_jobs')#,), pre_dispatch=None n_jobs=-1
#    reg=_GS.fit(X_train, y_train)
#    print(_GS.best_score_)
#    print(_GS.best_params_)
#



#    print("Metodo de tune: Randomized Search")
    #### solver 'lbfgs'
    #rs = RandomizedSearchCV(mlp, param_distributions={
        #'hidden_layer_sizes': [i for i in itertools.product(range(1,7),repeat=2)],
                  #'activation': ['relu', 'tanh', 'logistic'],
                  #'solver': ['lbfgs'],
                  #'power_t': [0.5],
                  #'alpha': np.linspace(0.001, 0.1, num=5),
                  #'max_iter': [1000, 3000, 5000],
                  #'early_stopping': [False],
                  #'warm_start': [False],
                  #'random_state': [9]})

    # {'warm_start': False, 'hidden_layer_sizes': (3, 4), 'activation': 'relu', 'max_iter': 3000, 'power_t': 0.5, 'random_state': 9, 'alpha': 0.075250000000000011, 'solver': 'lbfgs', 'early_stopping': False}
    ## solver 'adam'
#    rs = RandomizedSearchCV(mlp, param_distributions={ 'hidden_layer_sizes': [i for i in itertools.product(range(1,9),repeat=2)],
#                  'activation': ['relu', 'tanh', 'logistic'],
#                  'solver': ['adam','sgd'],
#                  'learning_rate_init':np.linspace(0.001, 0.1, num=10),
#                  'power_t': [0.5],
#                  'alpha': np.linspace(0.001, 0.1, num=10),
#                  'max_iter': [1000],
#                  'early_stopping': [True],
#                  'warm_start': [False],
#                  'random_state': [9]
#                  })

    ## solver 'sgd'
#    rs = RandomizedSearchCV(mlp, param_distributions={ 'hidden_layer_sizes': [i for i in itertools.product(range(1,9),repeat=2)],
##                  'activation': ['relu', 'tanh', 'logistic'],
#                  'activation': ['relu'],
#                  'solver': ['sgd'],
#                  'learning_rate': ['constant', 'invscaling', 'adaptive'],
##                  'learning_rate': ['adaptive'],
#                  'learning_rate_init':np.linspace(0.001, 0.1, num=10),
#                  'power_t': [0.5],
#                  'alpha': np.linspace(0.001, 0.1, num=10),
#                  'momentum': np.linspace(0.1, 0.9, num=10), # cuando se usa 'sgd'
#                  #'max_iter': [1000, 3000, 5000],
#                  'max_iter': [1000],
#                  'early_stopping': [True],
#                  'warm_start': [False],
#                  'random_state': [9]
#                  })

    # {'warm_start': False, 'solver': 'adam', 'activation': 'relu', 'max_iter': 5000, 'power_t': 0.5, 'random_state': 9, 'early_stopping': False, 'alpha': 0.001, 'learning_rate_init': 0.001, 'hidden_layer_sizes': (4, 3)}


    param={ 'hidden_layer_sizes': [i for i in itertools.product(range(4,12),repeat=2)],
#                  'activation': ['relu', 'tanh', 'logistic'],
                  'activation': ['relu'],
#                  'activation': ['logistic'],
                  'solver': ['sgd'],
#                  'learning_rate': ['constant'],##, 'invscaling', 'adaptive'],
                  'learning_rate': ['adaptive'],
                  'learning_rate_init':np.linspace(0.001, 0.1, num=5),
                  'power_t': [0.5],
                  'alpha': np.linspace(0.001, 0.1, num=5),
                  'momentum': np.linspace(0.1, 0.9, num=5), # cuando se usa 'sgd'
                  'max_iter': [1000],
                  'early_stopping': [True],
                  'validation_fraction': [0.25],
                  'warm_start': [False],
                  'random_state': [9]
                  }                  
    Scoring='neg_mean_squared_error'
    _RS = RandomizedSearchCV(mlp, param_distributions=param, scoring=Scoring,
                       cv=2, verbose=True)
          #, pre_dispatch='2*n_jobs')#,), pre_dispatch=None n_jobs=-1

#     {'warm_start': False, 'solver': 'adam', 'learning_rate': 'invscaling', 'max_iter': 1000, 'power_t': 0.5, 'random_state': 9, 'learning_rate_init': 0.067000000000000004, 'alpha': 0.023000000000000003, 'early_stopping': False, 'activation': 'relu', 'momentum': 0.54444444444444451, 'hidden_layer_sizes': (2, 3)}
    reg = _RS.fit(X_train, y_train)
    print(_RS.best_score_)
    print(_RS.best_params_)

#    print("Calibracion")
    y_pred_Cal = reg.predict(X_train)
#    print(y_pred_Cal)
    y_true_Cal = np.array(y_train)
#    print(y_true_Cal)
    rmse = statistics.RMSE(y_true_Cal, y_pred_Cal)
    print("RMSE Cal:" + str(rmse))
    data2 = pd.DataFrame({'yTest_SMAP' :y_true_Cal,'yAprox' :y_pred_Cal.T})
    RR = smf.ols('yTest_SMAP ~ 1+ yAprox', data2).fit().rsquared
    print("R^2 Cal: "+str(RR))


    print("Validacion")
    y_pred = reg.predict(X_test)
    y_true = np.array(y_test)
    rmse = statistics.RMSE(y_true, y_pred)
    print("RMSE Val:" + str(rmse))
    data2 = pd.DataFrame({'yTest_SMAP' :y_true,'yAprox' :y_pred.T})
    RR = smf.ols('yTest_SMAP ~ 1+ yAprox', data2).fit().rsquared
    print("R^2 Val: "+str(RR))

    return reg, np.array(y_pred_Cal), np.array(y_pred)
    
    
    
#####--------------------------------------------------------------------------

def tune_SAR_SMAP(xTraining,yTraining, xTest, yTest):


    mlp = MLPRegressor()
    ### solver 'lbfgs'
#    param_grid = {'hidden_layer_sizes': [i for i in itertools.product(range(1,7),repeat=2)],
#                  'activation': ['relu', 'tanh', 'logistic'],
#                  'solver': ['lbfgs'],
#                  'power_t': [0.5],
#                  'alpha': np.linspace(0.001, 0.1, num=5),
#                  'max_iter': [5000],
#                  'early_stopping': [False],
#                  'warm_start': [False],
#                  'random_state': [9]
#                  }
    ### solver 'adam'
#    param_grid = {'hidden_layer_sizes': [i for i in itertools.product(range(3,9),repeat=2)],
#                  'activation': ['relu', 'tanh', 'logistic'],
#                  'solver': ['adam'],
#                  'learning_rate_init':np.linspace(0.001, 0.1, num=5),
#                  'power_t': [0.5],
#                  'alpha': np.linspace(0.001, 0.1, num=5),
#                  'max_iter': [1000],
#                  'early_stopping': [False],
#                  'warm_start': [False],
#                  'random_state': [9]
#                  }
    #  {'warm_start': False, 'hidden_layer_sizes': (2, 3), 'activation': 'relu', 'max_iter': 1000, 'power_t': 0.5, 'random_state': 9, 'early_stopping': False, 'alpha': 0.001, 'solver': 'adam', 'learning_rate_init': 0.025750000000000002}
    ### solver 'sgd'
    param_grid = {'hidden_layer_sizes': [i for i in itertools.product(range(3,9),repeat=2)],
#                  'activation': ['relu', 'tanh', 'logistic'],
                  'activation': ['logistic'],
                  'solver': ['sgd'],
#                  'learning_rate': ['constant', 'invscaling', 'adaptive'],
                  'learning_rate': ['constant'],
                  'learning_rate_init':np.linspace(0.001, 0.1, num=5),
                  'power_t': [0.5],
                  'alpha': np.linspace(0.001, 0.1, num=5),
                  'momentum': np.linspace(0.1, 0.9, num=5), # cuando se usa 'sgd'
                  'max_iter': [100],
                  'early_stopping': [True],
                  'validation_fraction': [0.25],
                  'warm_start': [False],
                  'random_state': [9]
                  }

    Scoring='neg_mean_squared_error' 
    _GS = GridSearchCV(mlp, param_grid=param_grid, scoring=Scoring,
                       cv=2, verbose=True, pre_dispatch='2*n_jobs')
    
    reg=_GS.fit(xTraining,yTraining)

    print(_GS.best_score_)
    print(_GS.best_params_)



#    print("Metodo de tune: Randomized Search")
    #### solver 'lbfgs'
    #rs = RandomizedSearchCV(mlp, param_distributions={
        #'hidden_layer_sizes': [i for i in itertools.product(range(1,7),repeat=2)],
                  #'activation': ['relu', 'tanh', 'logistic'],
                  #'solver': ['lbfgs'],
                  #'power_t': [0.5],
                  #'alpha': np.linspace(0.001, 0.1, num=5),
                  #'max_iter': [1000, 3000, 5000],
                  #'early_stopping': [False],
                  #'warm_start': [False],
                  #'random_state': [9]})

    # {'warm_start': False, 'hidden_layer_sizes': (3, 4), 'activation': 'relu', 'max_iter': 3000, 'power_t': 0.5, 'random_state': 9, 'alpha': 0.075250000000000011, 'solver': 'lbfgs', 'early_stopping': False}
    ## solver 'adam'
#    rs = RandomizedSearchCV(mlp, param_distributions={ 'hidden_layer_sizes': [i for i in itertools.product(range(1,4),repeat=2)],
#                  'activation': ['relu', 'tanh', 'logistic'],
#                  'solver': ['adam','sgd'],
#                  'learning_rate_init':np.linspace(0.001, 0.1, num=10),
#                  'power_t': [0.5],
#                  'alpha': np.linspace(0.001, 0.1, num=10),
#                  'max_iter': [1000],
#                  'early_stopping': [True],
#                  'warm_start': [False],
#                  'random_state': [9]
#                  })

    ## solver 'sgd'
#    rs = RandomizedSearchCV(mlp, param_distributions={ 'hidden_layer_sizes': [i for i in itertools.product(range(1,4),repeat=2)],
#                  'activation': ['relu', 'tanh', 'logistic'],
##                  'activation': ['relu'],
#                  'solver': ['sgd'],
#                  'learning_rate': ['constant', 'invscaling', 'adaptive'],
##                  'learning_rate': ['adaptive'],
#                  'learning_rate_init':np.linspace(0.001, 0.1, num=10),
#                  'power_t': [0.5],
#                  'alpha': np.linspace(0.001, 0.1, num=10),
#                  'momentum': np.linspace(0.1, 0.9, num=10), # cuando se usa 'sgd'
#                  #'max_iter': [1000, 3000, 5000],
#                  'max_iter': [1000],
#                  'early_stopping': [True],
#                  'warm_start': [False],
#                  'random_state': [9]
#                  })

    # {'warm_start': False, 'solver': 'adam', 'activation': 'relu', 'max_iter': 5000, 'power_t': 0.5, 'random_state': 9, 'early_stopping': False, 'alpha': 0.001, 'learning_rate_init': 0.001, 'hidden_layer_sizes': (4, 3)}


#    rs = RandomizedSearchCV(mlp, param_distributions={ 'hidden_layer_sizes': [i for i in itertools.product(range(1,5),repeat=2)],
#                  'activation': ['relu', 'tanh', 'logistic'],
##                  'solver': ['sgd', 'adam', 'lbfgs'],
#                  'solver': ['sgd', 'adam'],
##                  'learning_rate': ['constant', 'invscaling', 'adaptive'],
#                  'learning_rate': ['adaptive'],
#                  'learning_rate_init':np.linspace(0.001, 0.1, num=10),
#                  'power_t': [0.5],
#                  'alpha': np.linspace(0.001, 0.1, num=10),
#                  'momentum': np.linspace(0.1, 0.9, num=10), # cuando se usa 'sgd'
#                  'max_iter': [2500],
#                  'early_stopping': [True],
#                  'warm_start': [False],
#                  'random_state': [9]
#                  })
#
###     {'warm_start': False, 'solver': 'adam', 'learning_rate': 'invscaling', 'max_iter': 1000, 'power_t': 0.5, 'random_state': 9, 'learning_rate_init': 0.067000000000000004, 'alpha': 0.023000000000000003, 'early_stopping': False, 'activation': 'relu', 'momentum': 0.54444444444444451, 'hidden_layer_sizes': (2, 3)}
#    reg = rs.fit(xTraining,yTraining)
#    print(rs.best_score_)
#    print(rs.best_params_)

#    print("Calibracion")

    y_pred_Cal = reg.predict(xTraining)
#    print(y_pred_Cal)
    y_true_Cal = np.array(yTraining)
#    print(y_true_Cal)
    rmse = statistics.RMSE(y_true_Cal, y_pred_Cal)
    print("RMSE Cal:" + str(rmse))
    data2 = pd.DataFrame({'yTest_SMAP' :y_true_Cal,'yAprox' :y_pred_Cal.T})
    RR = smf.ols('yTest_SMAP ~ 1+ yAprox', data2).fit().rsquared
    print("R^2 Cal: "+str(RR))


    print("Validacion")
    y_pred = reg.predict(xTest)
    y_true = np.array(yTest)
    rmse = statistics.RMSE(y_true, y_pred)
    print("RMSE Val:" + str(rmse))
    data2 = pd.DataFrame({'yTest_SMAP' :y_true,'yAprox' :y_pred.T})
    RR = smf.ols('yTest_SMAP ~ 1+ yAprox', data2).fit().rsquared
    print("R^2 Val: "+str(RR))

    return reg, np.array(y_pred_Cal), np.array(y_pred)
    
