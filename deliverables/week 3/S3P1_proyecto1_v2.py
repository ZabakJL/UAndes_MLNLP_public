#!/usr/bin/env python
# coding: utf-8

# ![image info](https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/images/banner_1.png)

# # Proyecto 1 - Predicción de precios de vehículos usados
# 
# En este proyecto podrán poner en práctica sus conocimientos sobre modelos predictivos basados en árboles y ensambles, y sobre la disponibilización de modelos. Para su desasrrollo tengan en cuenta las instrucciones dadas en la "Guía del proyecto 1: Predicción de precios de vehículos usados".
# 
# **Entrega**: La entrega del proyecto deberán realizarla durante la semana 4. Sin embargo, es importante que avancen en la semana 3 en el modelado del problema y en parte del informe, tal y como se les indicó en la guía.
# 
# Para hacer la entrega, deberán adjuntar el informe autocontenido en PDF a la actividad de entrega del proyecto que encontrarán en la semana 4, y subir el archivo de predicciones a la [competencia de Kaggle](https://www.kaggle.com/t/b8be43cf89c540bfaf3831f2c8506614).

# ## Datos para la predicción de precios de vehículos usados
# 
# En este proyecto se usará el conjunto de datos de Car Listings de Kaggle, donde cada observación representa el precio de un automóvil teniendo en cuenta distintas variables como: año, marca, modelo, entre otras. El objetivo es predecir el precio del automóvil. Para más detalles puede visitar el siguiente enlace: [datos](https://www.kaggle.com/jpayne/852k-used-car-listings).

# ## Ejemplo predicción conjunto de test para envío a Kaggle
# 
# En esta sección encontrarán el formato en el que deben guardar los resultados de la predicción para que puedan subirlos a la competencia en Kaggle.

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Importación librerías
import pandas as pd
import numpy as np
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[3]:


# Carga de datos de archivo .csv
dataTraining = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTrain_carListings.zip')
dataTesting = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTest_carListings.zip', index_col=0)


# In[4]:


# Visualización datos de entrenamiento
dataTraining


# In[5]:


# Visualización datos de test
dataTesting


# In[6]:


dataTraining.describe()


# In[7]:


varList1 = ['Year', 'State', 'Make', 'Model']
for column in varList1:
    set1 = set(list(dataTraining[column].unique()))
    set2 = set(list(dataTesting[column].unique()))

    # Obtener los valores que están en set1 pero no en set2
    valores_en_lista1_no_en_lista2 = set1 - set2

    # Obtener los valores que están en set2 pero no en set1
    valores_en_lista2_no_en_lista1 = set2 - set1

    print(f"Valores en dataTraining pero no en dataTesting para la columna {column}:", valores_en_lista1_no_en_lista2)
    print(f"Valores en dataTesting pero no en dataTraining para la columna {column}:", valores_en_lista2_no_en_lista1)


# In[8]:


correlaciones=dataTraining.corr(numeric_only=True)
correlaciones


# In[9]:


plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.scatterplot(y='Year', x='Price', data=dataTraining, marker='o', color='blue')
plt.title('Precio vs Año')

plt.subplot(1, 3, 2)
sns.scatterplot(y='Mileage', x='Price', data=dataTraining, marker='o', color='red')
plt.title('Precio vs Millas')

plt.subplot(1, 3, 3)
sns.scatterplot(x='Mileage', y='Year', data=dataTraining, marker='o', color='green')
plt.title('Años vs Millas')

plt.tight_layout()
plt.show()


# In[10]:


plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.boxplot(x=dataTraining['Price'], color='blue')
plt.title('Boxplot Price')

plt.subplot(1, 3, 2)
sns.boxplot(x=dataTraining['Mileage'], color='red')
plt.title('Boxplot Mileage')

plt.subplot(1, 3, 3)
sns.boxplot(x=dataTraining['Year'], color='green')
plt.title('Boxplot Year')

plt.tight_layout()
plt.show()


# In[11]:


def histogramas (data):
    melted = data.melt(var_name='variable', value_name='value')
    grid=sns.FacetGrid(melted, col='variable', col_wrap=3, sharex=False, sharey=False, height=4)

    # Mapear la función de histograma a cada celda de la cuadricula
    grid.map(sns.histplot, 'value', kde=True, orientation='horizontal')

    # Ajustar el diseño y mostrar la figura
    plt.tight_layout()
    plt.show()


# In[12]:


histogramas(dataTraining[['Price', 'Year', 'Mileage']])


# In[13]:


######################################################
### Preparar datos
######################################################

def prepareData (df, isTesting=False, colsList=[]):
    # Concatenar las variables Make y Model
    df['Make_Model']=df['Make']+'_'+df['Model'] 

    # Crear la lista de variables según el conjunto de datos que se esta transformando
    if isTesting == False:
        varList1 = ['Price', 'Year', 'Mileage', 'State', 'Make_Model']
        varList2 = ['Price', 'Year', 'Mileage']
    else:
        varList1 = ['Year', 'Mileage', 'State', 'Make_Model']
        varList2 = ['Year', 'Mileage']

    # Filtrar el dataframe por las columnas de interes
    df = df[varList1]
    
    # Crear la instancia de OneHotEncoder y las columnas dummy para State y Make_Model
    encoder = OneHotEncoder(drop='first', sparse=False) # Usamos drop=’first’ para eliminar la primera categoría en cada característica
    colsToEncoded=['State', 'Make_Model']
    dfCoded = pd.DataFrame(encoder.fit_transform(df[colsToEncoded]))
    # nombrar las columnas dummy
    dfCoded.columns = encoder.get_feature_names_out(colsToEncoded)
    # agregar las columnas 'Year', 'Mileage' o 'Price', 'Year', 'Mileage' según corresponda al conjunto de datos
    dfCoded[varList2]=df[varList2]

    if isTesting==True:
        columnas_faltantes = set(colsList) - set(dfCoded.columns)
        for columna in columnas_faltantes:
            dfCoded[columna] = 0
    
    dfCoded = dfCoded.sort_index(axis=1)
    return dfCoded


# In[14]:


# Transformar datos y crear dummies en train
dataTrainingCoded = prepareData(dataTraining.copy())
# Separar predictores y resultado
XTotalTrain = dataTrainingCoded.drop(columns=['Price'])
yTotalTrain = dataTrainingCoded[['Price']]
# Transformar datos y crear dummies en test
dataTestCoded = prepareData(dataTraining.copy(), True, XTotalTrain.columns)


# In[15]:


print(XTotalTrain.shape)
print(dataTestCoded.shape)


# In[16]:


# Crear y entrenar el modelo
model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree = 0.4, gamma = 900000.0, learning_rate = 0.3, max_depth = 9, n_estimators = 600, random_state=0)
model.fit(XTotalTrain, yTotalTrain)
# y_pred = model.predict(X_Test)


# In[17]:


# Exportar modelo a archivo binario .pkl
import joblib
import os
#os.chdir('..')
joblib.dump(model, 'api/usedCarPrices.pkl', compress=3) # La ruta antes del nombre del archivo binario debe existir


# In[18]:


os.getcwd()


# In[19]:


# Función para importar el modelo y predecir
predictors = XTotalTrain.columns

def usedCarPricesPredict(Year, Mileage, State, Make, Model):
    clf = joblib.load('api/usedCarPrices.pkl') 
    data_ = pd.DataFrame({'Year': [Year], 'Mileage': [Mileage], 'State': [State], 'Make': [Make], 'Model': [Model]})
    X_Test = prepareData(data_, True, predictors)
    # Make prediction
    p1 = clf.predict(X_Test)
    return p1


# In[20]:


# Predicción de probabilidad de que un link sea phishing
usedCarPricesPredict(2014, 31909, 'MD', 'Nissan', 'MuranoAWD')


# In[25]:


# Predicción de probabilidad de que un link sea phishing
usedCarPricesPredict(2017, 5362, 'FL', 'Jeep', 'Wrangle')
#2017	5362	FL	Jeep	Wrangle


# In[26]:


# Importación librerías
from flask import Flask
from flask_restx import Api, Resource, fields


# In[27]:


# Definición aplicación Flask
app = Flask(__name__)

# Definición API Flask
api = Api(
    app, 
    version='1.0', 
    title='Used Car Price Prediction API',
    description='used car price prediction API')

ns = api.namespace('predict', 
     description='price prediction')

# Definición argumentos o parámetros de la API
parser = api.parser()
parser.add_argument(
    'Year', 
    type=int, 
    required=True, 
    help='year of manufacture', 
    location='args')

parser.add_argument(
    'Mileage', 
    type=float, 
    required=True, 
    help='vehicle mileage', 
    location='args')

parser.add_argument(
    'State', 
    type=str, 
    required=True, 
    help='Abbreviation of US state names', 
    location='args')

parser.add_argument(
    'Make', 
    type=str, 
    required=True, 
    help='vehicle manufacturer', 
    location='args')

parser.add_argument(
    'Model', 
    type=str, 
    required=True, 
    help='vehicle model', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})


# In[28]:


# Definición de la clase para disponibilización
@ns.route('/')
class UsedCarPricePredictionAPI(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()        
        prediction = usedCarPricesPredict(args['Year'], args['Mileage'], args['State'], args['Make'], args['Model'])
        return {"result": prediction[0]}, 200
        #return jsonify({'predicted_price': prediction[0]})
        #return jsonify({'predicted_price': args})


# In[29]:


# Ejecución de la aplicación que disponibiliza el modelo de manera local en el puerto 5000
app.run(debug=True, use_reloader=False, host='0.0.0.0', port=88)


# In[ ]:




