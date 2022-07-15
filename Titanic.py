# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#librerias a usar
import numpy as np
import pandas as pd


#importar datos
url_test = '/Users/Edmundo/Downloads/Python/test.csv'
url_train ='/Users/Edmundo/Downloads/Python/train.csv'

df_test = pd.read_csv(url_test)
df_train = pd.read_csv(url_train)

print(df_test.head())
print(df_train.head())

#verifico el tipo de contenido en los datos
print('tipos de datos')
print(df_train.info())
print(df_test.info())

#verifico los datos faltantes en el dataset
print('datos faltantes')
print(pd.isnull(df_train).sum())
print(pd.isnull(df_test).sum())

#verifico las estadisticas del dataset
print('estaditicas dataset')
print(df_train.describe())
print(df_test.describe())


#preprocesamiento de ls datos
#cambio de los datos de sexo a numeros
df_train['Sex'].replace(['female','male'],[0,1],inplace=True)
df_test['Sex'].replace(['female','male'],[0,1],inplace=True)

#cambio de los datos embarque a numeros
df_train['Embarked'].replace(['Q','S','C'],[0,1,2],inplace=True)
df_test['Embarked'].replace(['Q','S','C'],[0,1,2],inplace=True)

#reemplazo de los datos faltantes en la edad por la media de esta columna
print(df_train['Age'].mean())
print(df_test['Age'].mean())

promedio=(df_train['Age'].mean() + df_test['Age'].mean())/2
#promedio=30

df_train['Age']=df_train['Age'].replace(np.nan, promedio)
df_test['Age']=df_test['Age'].replace(np.nan, promedio)

#creo varios grupos de acuerdo a las bandas de edades
#bandas 0-8,9-15,16-18,19-25,26-40,41-60,61-100
bins=[0,8,15,18,25,40,60,100]
names=['1','2','3','4','5','6','7']
df_train['Age']=pd.cut(df_train['Age'],bins,labels=names)
df_test['Age']=pd.cut(df_test['Age'],bins,labels=names)

#se elimina la columna cabin  ya que tiene muchos datos perdido
df_train.drop(['Cabin'],axis=1,inplace=True)
df_test.drop(['Cabin'],axis=1,inplace=True)

#elimino las columnas que consdiero no son necesarias para el ejercicio
df_train=df_train.drop(['PassengerId','Name','Ticket'],axis=1)
df_test=df_test.drop(['Name','Ticket'],axis=1)


#se eliminan las filas con los datos perdidos
df_train.dropna(axis=0, how='any',inplace=True)
df_test.dropna(axis=0, how='any',inplace=True)


#verifico los datos
print(pd.isnull(df_train).sum())
print(pd.isnull(df_test).sum())
print()
print(df_train.shape)
print(df_test.shape)
print()
print(df_test.head())
print(df_train.head())


#######algoritmos de ML#########
#librerias a utilizar
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

#separo la columna con la informacion de los sobrevivientes
X = np.array(df_train.drop(['Survived'],1))
y = np.array(df_train['Survived'])


#separo los datos de "train" una parte para entrenamiento y otra para prueba
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


#regresion logistica
logreg= LogisticRegression()
logreg.fit(X_train, y_train)
Y_pred= logreg.predict(X_test)
print('PRECISION REGRESION LOGISTICA: ')
print(logreg.score(X_train, y_train))


#support vector machines
svc = SVC()
svc.fit(X_train, y_train)
Y_pred =svc.predict(X_test)
print('PRECISION SOPORTE DE VECTORES: ')
print(svc.score(X_train, y_train))

#k neighbors
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
print('PRECISION VECINOS MAS CERANOS: ')
print(knn.score(X_train, y_train))


#########Prediccion usando los modelos##########
#
ids=df_test['PassengerId']

#regresion logistica
prediccion_logreg = logreg.predict(df_test.drop('PassengerId',axis=1))
out_logreg=pd.DataFrame({'PassengerID':ids,'Survived':prediccion_logreg})
print('Prediccion regresion logistica: ')
print(out_logreg.head())


#sporte de vectores
prediccion_svc = svc.predict(df_test.drop('PassengerId',axis=1))
out_svc=pd.DataFrame({'PassengerId':ids, 'Survived':prediccion_svc})
print('PREDICCION SVC: ')
print(out_svc.head())

#kneighbors
prediccion_knn=knn.predict(df_test.drop('PassengerId',axis=1))
out_knn=pd.DataFrame({'PassengerId':ids, 'Survived':prediccion_knn})
print('PREDICCION knn: ')
print(out_knn.head())

