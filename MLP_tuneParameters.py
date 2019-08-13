
# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Wed Oct 24 10:16:04 2018

@author: gag 

Script that allows you to obtain the architecture and parameters of the neural network,
that is, it allows you to find the appropriate number of hidden layers and neurons of them,
together with key parameters of the network such as: type of activation function, speed of 
learning, penalty , among others. Samples of parameters:

                'hidden_layer_sizes': [i for i in itertools.product(range(3,9),repeat=2)]
                'solver': ['lbfgs', 'adam', sgd'],
                The solver are:
                - 'lbfgs' is an optimizer in the family of quasi-Newton methods.
                - 'sgd' refers to stochastic gradient descent.
                - 'adam' refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba

                'activation': ['relu', 'tanh', 'logistic'],
                'learning_rate_init':np.linspace(0.001, 0.1, num=5),
                'power_t': [0.5],
                'alpha': np.linspace(0.001, 0.1, num=5),
                'max_iter': [1000],
                'early_stopping': [False],
                'warm_start': [False],
                'random_state': [9

Each solver method has its set of parameters so it separates them. 

To carry out this search in the space of hyperparameters, optimization methods are used, 
those tested in this script are:
    * Exhaustive Grid Search
    * Randomized Parameter Optimization

"""

import numpy as np
import pandas as pd
import statistics


from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
import statsmodels.formula.api as smf




def tune_Grid_Search(xTraining,yTraining, xTest, yTest, solverType):
    """
    Exhaustive Grid Search
    """
    mlp = MLPRegressor()
    ### solver 'lbfgs'
    if(solverType == 'lbfgs'):
        param_grid = {'hidden_layer_sizes': [i for i in itertools.product(range(1,7),repeat=2)],
                        'activation': ['relu', 'tanh', 'logistic'],
                        'solver': ['lbfgs'],
                        'power_t': [0.5],
                        'alpha': np.linspace(0.001, 0.1, num=5),
                        'max_iter': [5000],
                        'early_stopping': [False],
                        'warm_start': [False],
                        'random_state': [9]
                        }

    ### solver 'adam'
    if(solverType == 'adam'):
        param_grid = {'hidden_layer_sizes': [i for i in itertools.product(range(3,9),repeat=2)],
                        'activation': ['relu', 'tanh', 'logistic'],
                        'solver': ['adam'],
                        'learning_rate_init':np.linspace(0.001, 0.1, num=5),
                        'power_t': [0.5],
                        'alpha': np.linspace(0.001, 0.1, num=5),
                        'max_iter': [1000],
                        'early_stopping': [False],
                        'warm_start': [False],
                        'random_state': [9]
                        }

    ### solver 'sgd'
    if(solverType == 'sgd'):
        param_grid = {'hidden_layer_sizes': [i for i in itertools.product(range(3,9),repeat=2)],
                    'activation': ['relu', 'tanh', 'logistic'],
                    # 'activation': ['logistic'],
                    'solver': ['sgd'],
                    'learning_rate': ['constant', 'invscaling', 'adaptive'],
                    # 'learning_rate': ['constant'],
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

    print("Calibracion")
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
    



def tune_Randomized_Search(xTraining,yTraining, xTest, yTest, solverType):
    """
    Randomized Parameter Optimization
    """
    mlp = MLPRegressor()

    #### solver 'lbfgs'
    if(solverType == 'lbfgs'):
        param_distributions={'hidden_layer_sizes': [i for i in itertools.product(range(1,7),repeat=2)],
                            'activation': ['relu', 'tanh', 'logistic'],
                            'solver': ['lbfgs'],
                            'power_t': [0.5],
                            'alpha': np.linspace(0.001, 0.1, num=5),
                            'max_iter': [1000, 3000, 5000],
                            'early_stopping': [False],
                            'warm_start': [False],
                            'random_state': [9]}

    ## solver 'adam'
    if(solverType == 'adam'):
        param_distributions={ 'hidden_layer_sizes': [i for i in itertools.product(range(1,4),repeat=2)],
                            'activation': ['relu', 'tanh', 'logistic'],
                            'solver': ['adam','sgd'],
                            'learning_rate_init':np.linspace(0.001, 0.1, num=10),
                            'power_t': [0.5],
                            'alpha': np.linspace(0.001, 0.1, num=10),
                            'max_iter': [1000],
                            'early_stopping': [True],
                            'warm_start': [False],
                            'random_state': [9]
                            }

    ## solver 'sgd'
    if(solverType == 'sgd'):
        param_distributions={ 'hidden_layer_sizes': [i for i in itertools.product(range(1,4),repeat=2)],
                            'activation': ['relu', 'tanh', 'logistic'],
                            #'activation': ['relu'],
                            'solver': ['sgd'],
                            'learning_rate': ['constant', 'invscaling', 'adaptive'],
                            #'learning_rate': ['adaptive'],
                            'learning_rate_init':np.linspace(0.001, 0.1, num=10),
                            'power_t': [0.5],
                            'alpha': np.linspace(0.001, 0.1, num=10),
                            'momentum': np.linspace(0.1, 0.9, num=10), # cuando se usa 'sgd'
                            #'max_iter': [1000, 3000, 5000],
                            'max_iter': [1000],
                            'early_stopping': [True],
                            'warm_start': [False],
                            'random_state': [9]
                            }

    Scoring='neg_mean_squared_error' 
    _RS = RandomizedSearchCV(mlp, param_distributions, scoring=Scoring,
                            cv=2, verbose=True, pre_dispatch='2*n_jobs')

    reg = _RS.fit(xTraining,yTraining)

    print(_RS.best_score_)
    print(_RS.best_params_)

    print("Calibracion")
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