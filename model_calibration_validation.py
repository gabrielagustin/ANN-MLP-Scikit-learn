# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Wed Oct 24 10:16:04 2018
@author: gag 

Script that receives a .CSV file that has a table with the input variables next to the output variable.
The Perceptron Multilayer (MLP) as a regressor is applied to the data set.

"""


import pandas as pd
import selection
import matplotlib.pyplot as plt
import numpy as np
import statistics
import lectura
import MLP
import seaborn as sns


file = "/media/gag/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/mediciones_sensores_CONAE_MonteBuey_SMAP/SM_CONAE_Prom/extract_table_2.csv"

data = lectura.lecturaCompleta_etapa1_SAR_SMAP(file)
#print data

#### the observations in the table are mixed
rand = 0
np.random.seed(rand)
dataNew = selection.shuffle(data)
dataNew = dataNew.reset_index(drop=True)

#### the data set is divided for training and testing
dataTraining, dataTest = train_test_split(data, test_size=0.25)

print("--------------------------------------------------------")
print("Statistics training data")
print(dataTraining.describe())
print("--------------------------------------------------------")
print("Statistics test data")
print(dataTest.describe())
print("--------------------------------------------------------")

#### MLP training
MLPmodel, yCalMLP = MLP.mlp_SAR_SMAP(dataTraining)

yTraining = dataTraining['SM_CONAE']
yTraining = 10**(yTraining)

df = pd.DataFrame({'yTraining':yTraining,
                   'yCalMLP':yCalMLP
                   })


fig = plt.figure(1,facecolor="white")
ax = fig.add_subplot(111)
ax.set_xlim(5,50)
ax.set_ylim(5,50)
sns.regplot(x="yTraining", y="yCalMLP", fit_reg=True, data=df, scatter_kws={'s':50}, color="g", label='MLP', ax=ax)
plt.xlabel('Observed value [% Vol.]', fontsize=12);
plt.ylabel('Estimated value [% Vol.]', fontsize=12);
#plt.title('Training');
# Move the legend to an empty part of the plot
plt.legend(loc='lower right')


#### Test MLP

yTest = np.array(dataTest["SM_CONAE"])
yTest = 10**(yTest)
del dataTest["SM_CONAE"]
xTest = dataTest

yAproxMLP = MLPmodel.predict(xTest)

df = pd.DataFrame({'y':yTest,
                   'yAproxMLP':yAproxMLP
                   })

#dataNew = df[(df.y < 44)]
#df = dataNew

fig = plt.figure(2,facecolor="white")
ax = fig.add_subplot(111)
ax.set_xlim(5,50)
ax.set_ylim(5,50)
sns.regplot(x="y", y="yAproxMLP", fit_reg=True, data=df, scatter_kws={'s':50}, color="g", label='MLP', ax=ax)
plt.xlabel('Observed value [% Vol.]', fontsize=12);
plt.ylabel('Estimated value [% Vol.]', fontsize=12);
#plt.title('Test');
# Move the legend to an empty part of the plot
plt.legend(loc='lower right')
plt.grid(True)

plt.show()


