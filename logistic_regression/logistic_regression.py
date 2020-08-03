#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 18:28:37 2020

@author: cibelesantos

Análise de Dados de Imóveis nos EUA utilizando Regressão Log'sitica

No modelo de regressão linear, a variável dependente é considerada contínua, 
enquanto na regressão logística é categórica, ou seja, discreta. 
Na aplicação, o primeiro é usado em configurações de regressão, 
enquanto o último é usado para classificação binária ou multi-classe 
(onde é chamado de regressão logística multinomial)

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#-----------------------------------------------------------
#Importação do arquivo
#-----------------------------------------------------------

diretorio = '/Users/cibelesantos/Repos/machine-learning-methods/logistic_regression/'

df_advertising = pd.read_csv(diretorio + 'advertising.csv')


df_advertising.info()

df_advertising.describe()

df_advertising.columns
df_advertising.columns = ['daily_time_spent_on_site', 'age', 'area_income',
       'daily_internet_usage', 'ad_topic_line', 'city', 'male', 'country',
       'timestamp', 'clicked_on_ad']

#-----------------------------------------------------------
#Exploração do Dataset
#-----------------------------------------------------------

sns.set_style('whitegrid')

#Distribuição da idade
df_advertising['age'].hist(bins=30)
plt.xlabel('age')

sns.jointplot(x='age',y='area_income',data=df_advertising)


sns.jointplot(x='age',y='daily_time_spent_on_site',data=df_advertising,color='red',kind='kde');

sns.jointplot(x='daily_time_spent_on_site',y='daily_internet_usage',
              data=df_advertising,color='green')


sns.pairplot(df_advertising,hue='clicked_on_ad',palette='bwr')


#-----------------------------------------------------------
#Treinamento do modelo
#-----------------------------------------------------------
X = df_advertising[['daily_time_spent_on_site', 'age', 'area_income',
                    'daily_internet_usage', 'male']]
y = df_advertising['clicked_on_ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

score = logmodel.score(X_test, y_test)
score

#-----------------------------------------------------------
#Predições do modelo
#-----------------------------------------------------------

predictions = logmodel.predict(X_test)

print(classification_report(y_test,predictions))

cm = metrics.confusion_matrix(y_test, predictions)


plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".2f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);

























