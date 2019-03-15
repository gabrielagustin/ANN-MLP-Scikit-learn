# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Wed Oct 24 10:16:04 2018
@author: gag 

Script que recibe el archivo .CSV que posee una tabla con las variables de entrada
junto a la variable de salida. 
Al conjunto de datos le aplica el modelo: 
 - Perceptron Multicapa (MLP) como regresor
 

"""



import pandas as pd
import statsmodels.formula.api as smf
import selection
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statistics
import sklearn
import lectura
import MLPregression
import Mars
import copy
import SMmaps
import seaborn as sns



file = "/media/gag/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/mediciones_sensores_CONAE_MonteBuey_SMAP/SM_CONAE_Prom/extract_table_2.csv"



data = lectura.lecturaCompleta_etapa1_SAR_SMAP(file)
#print data

#file_name = "/home/gag/Escritorio/tabla1.csv"
#data.to_csv(file_name)


## se mezclan las observaciones de las tablas
## semilla para mezclar los datos en forma aleatoria

rand = 0
np.random.seed(rand)
dataNew = selection.shuffle(data)
dataNew = dataNew.reset_index(drop=True)

## formula

formula = "SM_CONAE ~ 1+Ts_SMAP+GPM+sigma0_vv_"

print("Modelo planteado:" + str(formula))
model2 = smf.ols(formula, dataNew).fit()
print("R^2 del modelo: " + str(model2.rsquared))


####### obtencion automatica de los porcentajes de datos
###### para entrenar y probar

print("Obtencion automatica de los porcentajes de datos")
type = "RMSE"
#type = "R^2"
var = "RSOILMOIST"
porc = 0.75
print("Porcentaje de datos de calculo: " + str(porc))


#### division de los datos para entrenamiento y prueba
nRow = len(dataNew.index)
numTraining=int(round(nRow)*porc)
print("Cantidad de elementos para el calculo de coeff: " + str(numTraining))
numTest=int((nRow)-numTraining)
print("Cantidad de elementos para prueba: " +str(numTest))


dataTraining =  dataNew.ix[:numTraining, :]
dataTraining = selection.shuffle(dataTraining)
dataTraining = dataTraining.reset_index(drop=True)


#lectura.graph2(dataTraining)


dataTest = dataNew.ix[numTraining + 1:, :]



print("-------------------------AQUIII------------------------------------------")
print(dataTraining)
print(dataTraining.describe())


#dataTraining, dataTest = train_test_split(data, test_size=0.25)
oPanda = dataTraining.copy(deep=False)

#print dataTest
#lectura.graph(dataTest)

#### Calibracion
print("Calibracion: ")
MLRmodel = smf.ols(formula, dataTraining).fit()
print(MLRmodel.summary())
print("R^2 del modelo: " + str(MLRmodel.rsquared))


#### error de calibracion
xxx = copy.copy(dataTraining)
del xxx['SM_CONAE']
yTraining = dataTraining['SM_CONAE']
yCal = MLRmodel.predict(xxx)
yTraining = 10**(yTraining)
yCal = 10**(yCal)

rmse = statistics.RMSE(np.array(yTraining),np.array(yCal))
print("RMSE:" + str(rmse))
bias = statistics.bias(yTraining,yCal)
print("Bias:" + str(bias))


#print "RMSE del modelo: " + str(np.sqrt(model.mse_resid))

#### se guardan los coeficientes del modelo entrenado
print("Los coeficientes del modelo son: ")
coeff =  MLRmodel.params
#print coeff[1]

print("Calculo de los VIF: ")
print("Orden de las variables")
print(list(dataNew))
matrix = np.array(dataTraining)
vifs = statistics.calc_vif(matrix)
print(vifs)
#vifs = aca.variance_inflation_factor(matrix,2)
#print vifs
#### prueba del modelo

print("Validacion: ")

y = np.array(dataTest["SM_CONAE"])

pred = MLRmodel.predict(dataTest)
yAprox = np.array(pred)
#### aca!!!
#y = np.exp(y)
#yAprox = np.exp(yAprox)

y = 10**(y)
yAprox = 10**(yAprox)

bias = statistics.bias(yAprox,y)
print("Bias Validacion:" + str(bias))

print("Rango real: "+ str(np.max(y))+"..." + str(np.min(y)))
print("Rango aproximado: "+ str(np.max(yAprox))+"..." + str(np.min(yAprox)))


## se obtiene el error
rmse = 0
rmse = statistics.RMSE(y,yAprox)
print("RMSE:" + str(rmse))
RR = sklearn.metrics.r2_score(y, yAprox)
print("R2:" + str(RR))
error = np.zeros((len(y),1))
#for i in range(0,len(error)):
error = np.abs(y-yAprox)


d = {'y': y, 'yAprox': yAprox}
df = pd.DataFrame(data=d)
RR = smf.ols('y ~ 1+ yAprox', df).fit().rsquared
print("R^2 222: "+str(RR))


sns.set_style("whitegrid")


#### se calibra metodo MLP
MLPmodel, yCalMLP, yAproxMLP = MLPregression.mlp_SAR_SMAP(porc,file, "etapa1", rand)



#### se calibra metodo MARS
MARSmodel, yCalMARS, yAproxMARS = Mars.mars_SAR_SMAP(porc, file, rand)

v1 = []
v2 = []
v3 = []
v4 = []
for i in range(len(yTraining)):
    v1.append(float(yTraining[i]))
    v2.append(float(yCal[i]))
    v3.append(float(yCalMLP[i]))
    v4.append(float(yCalMARS[i]))

#fig = plt.figure(1,facecolor="white")
#fig1 = fig.add_subplot(111,aspect='equal')
##fig1, ax = plt.subplots()


df = pd.DataFrame({'yTraining':yTraining,
                   'yCal':yCal,
                   'yCalMLP':yCalMLP,
                   'yCalMARS':yCalMARS
                   })

#df.loc[df.yCalMLP==1, 'yCal'] *= 2

#sns.pairplot(data=df,
            #x_vars=['yTraining'],
            #y_vars=['yCal', 'yCalMLP'])

fig = plt.figure(1,facecolor="white")
ax = fig.add_subplot(111)
ax.set_xlim(5,50)
ax.set_ylim(5,50)
sns.regplot(x="yTraining", y="yCal", marker="+", fit_reg=True, data=df, scatter_kws={'s':50}, label='MLR', ax=ax)
sns.regplot(x="yTraining", y="yCalMLP", fit_reg=True, data=df, scatter_kws={'s':50}, color="g", label='MLP', ax=ax)
sns.regplot(x="yTraining", y="yCalMARS", marker="x", fit_reg=True, data=df, scatter_kws={'s':50}, color="r", label='MARS', ax=ax)
plt.xlabel('Observed value [% Vol.]', fontsize=12);
plt.ylabel('Estimated value [% Vol.]', fontsize=12);
#plt.title('Scatterplot for the Association between Breast Cancer and Female Employment');
# Move the legend to an empty part of the plot
plt.legend(loc='lower right')





#plt.rcParams['figure.figsize'] = (20.0, 10.0)
#plt.rcParams['font.family'] = "serif"
#plt.show()

##fig1.set_title('Humedad Vs Humedad Aprox')
#fig1.scatter(yTraining ,yCal,  color='black', s=30, label='MLR')
#fig1.scatter(yTraining,yCalMLP, marker='x', color="green", s=40,  label='MLP')
#fig1.scatter(yTraining,yCalMARS, marker="+", color="blue", s=40, label='MARS')
#yt =np.array(yTraining)

#z = np.polyfit(v1,v2, 1)
#g = np.poly1d(z)
#fig1.plot(v1,g(v1),'black')

#z = np.polyfit(v1,v3, 1)
#g = np.poly1d(z)
#fig1.plot(v1,g(v1),'green')

#z = np.polyfit(v1,v4, 1)
#g = np.poly1d(z)
#fig1.plot(v1,g(v1),'blue')




#fig1.set_xlabel("observed value [% Vol.]",fontsize=12)
#fig1.set_ylabel("estimated value [% Vol.]",fontsize=12)
##fig1.set_xlabel("Valor observado [% Vol.]",fontsize=12)
##fig1.set_ylabel("Valor estimado [% Vol.]",fontsize=12)



##fig1.plot(yt, yt)
#fig1.legend(loc=4, fontsize = 'medium')
#fig1.axis([5,45, 5,45])

#x = np.linspace(*fig1.get_xlim())
#fig1.plot(x, x, linestyle="--")

#plt.grid(True)

df = pd.DataFrame({'y':y,
                   'yAprox':yAprox,
                   'yAproxMLP':yAproxMLP,
                   'yAproxMARS':yAproxMARS
                   })



#dataNew = df[(df.y < 44)]
#df = dataNew


fig = plt.figure(2,facecolor="white")
ax = fig.add_subplot(111)
ax.set_xlim(5,50)
ax.set_ylim(5,50)
sns.regplot(x="y", y="yAprox", marker="+", fit_reg=True, data=df, scatter_kws={'s':50}, label='MLR', ax=ax)
sns.regplot(x="y", y="yAproxMLP", fit_reg=True, data=df, scatter_kws={'s':50}, color="g", label='MLP', ax=ax)
sns.regplot(x="y", y="yAproxMARS", marker="x", fit_reg=True, data=df, scatter_kws={'s':50}, color="r", label='MARS', ax=ax)
plt.xlabel('Observed value [% Vol.]', fontsize=12);
plt.ylabel('Estimated value [% Vol.]', fontsize=12);
#plt.title('Scatterplot for the Association between Breast Cancer and Female Employment');
# Move the legend to an empty part of the plot
plt.legend(loc='lower right')
plt.grid(True)






plt.show()




##fig = plt.figure(2,facecolor="white")
##fig1 = fig.add_subplot(111, aspect='equal')

###fig1, ax = plt.subplots()

###fig1.set_title('Humedad Vs Humedad Aprox')
##fig1.scatter(y,yAprox, s=10, color='black',linewidth=3, label='MLR')
##fig1.scatter(y,yAproxMLP, s=10, marker="^", color="green", linewidth=3, label='MLP')
##fig1.scatter(y,yAproxMARS, s=10, marker="*", color="blue", linewidth=3, label='MARS')






##v1 = []
##v2 = []
##v3 = []
##v4 = []
##for i in range(len(y)):
    ##v1.append(float(y[i]))
    ##v2.append(float(yAprox[i]))
    ##v3.append(float(yAproxMLP[i]))
    ##v4.append(float(yAproxMARS[i]))


##z = np.polyfit(v1,v2, 1)
##g = np.poly1d(z)
##fig1.plot(v1,g(v1),'black')

##z = np.polyfit(v1,v3, 1)
##g = np.poly1d(z)
##fig1.plot(v1,g(v1),'green')

##z = np.polyfit(v1,v4, 1)
##g = np.poly1d(z)
##fig1.plot(v1,g(v1),'blue')


##cor = np.corrcoef(v1,v2)[0,1]
##if (cor >0 ):
    ##cor=(cor)*(cor)
##else:
    ##cor=(cor*(-1))*(cor*(-1))

##fig1.text(np.min(v1), np.max(v2), 'r^2=%5.3f' % cor, fontsize=15)

##fig1.text(16, 35, 'R^2=%5.3f' % RR, fontsize=12)
##fig1.text(16, 32, 'r^2=%5.3f' % cor, fontsize=12)

#fig1.set_xlabel("observed value [% Vol.]",fontsize=12)
#fig1.set_ylabel("estimated value [% Vol.]",fontsize=12)

##fig1.set_xlabel("Valor observado [% Vol.]",fontsize=12)
##fig1.set_ylabel("Valor estimado [% Vol.]",fontsize=12)

#fig1.legend(loc=4, fontsize = 'medium')
#fig1.axis([5,45, 5,45])

#x = np.linspace(*fig1.get_xlim())
#fig1.plot(x, x, linestyle="--")

#plt.grid(True)


##print "Correlacion de Pearson"+":" + str(cor)
##print "R^2 de la prueba"+":" + str(RR)

#plt.show()

##print "Aplication: "
##file = "tabla_aplication.csv"
##file = "tabla_calibration_validation.csv"
##application.application(file, model, MLPmodel, "etapa1")


print(data.describe())
print("Presionar tecla para continuar")
input()

## Se obtienen los mapas de HS con los modelos calibrados
SMmaps.calculateMaps(MLRmodel, MLPmodel, MARSmodel, "etapa1")
#
