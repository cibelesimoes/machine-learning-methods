#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 15:53:34 2020

@author: cibelesantos

Análise de Dados de Imóveis nos EUA utilizando Regressão Linear

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#-----------------------------------------------------------
#Importação do arquivo
#-----------------------------------------------------------

diretorio = '/Users/cibelesantos/Repos/machine-learning-methods/linear_regression/'

df_houses = pd.read_csv(diretorio + 'USA_Housing.csv')

df_houses.columns
df_houses.columns = ['avg_area_income', 'avg_area_house_age', 'avg_area_number_of_rooms',
       'avg_area_number_of_bedrooms', 'avg_area_population', 'price', 'address']


df_houses.info()
df_houses.describe()

#-----------------------------------------------------------
#Exploração dos dados
#-----------------------------------------------------------

sns.pairplot(df_houses)

sns.distplot(df_houses['price'])

sns.heatmap(df_houses.corr())

#-----------------------------------------------------------
#Treinamento do modelo
#-----------------------------------------------------------

X = df_houses[['avg_area_income', 'avg_area_house_age', 'avg_area_number_of_rooms',
               'avg_area_number_of_bedrooms', 'avg_area_population']]
y = df_houses['price']

#split treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


#Criando e treinando o modelo
lm = LinearRegression()
lm.fit(X_train,y_train)

# Printando a interceção: ou seja, quando que a "reta" atingirá o y
lm.intercept_

coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['coefficient'])
coeff_df


#-----------------------------------------------------------
#Predições do modelo
#-----------------------------------------------------------

predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)

sns.distplot((y_test-predictions),bins=50)


#-----------------------------------------------------------
#Medindo o modelo
#-----------------------------------------------------------
#Mean absolute error: isso significa que quando o modelo erra, ele erra em 82288.22 dólares
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

#Mean Squared Error: MSE é mais popular que o MAE, porque a MSE "puniria" erros maiores, 
#o que tende a ser útil no mundo real.

print('MSE:', metrics.mean_squared_error(y_test, predictions))

#Root Mean Square Error: RMSE é ainda mais popular do que MSE, porque o RMSE é 
#interpretável nas unidades "y".
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))














