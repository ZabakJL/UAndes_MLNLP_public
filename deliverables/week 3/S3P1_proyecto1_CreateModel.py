import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import xgboost as xgb


from sklearn.preprocessing import OneHotEncoder

import joblib

# Carga de datos de archivo .csv
dataTraining = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTrain_carListings.zip')
dataTesting = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTest_carListings.zip', index_col=0)

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

# Transformar datos y crear dummies en train
dataTrainingCoded = prepareData(dataTraining.copy())
# Separar predictores y resultado
XTotalTrain = dataTrainingCoded.drop(columns=['Price'])
yTotalTrain = dataTrainingCoded[['Price']]
# Transformar datos y crear dummies en test
dataTestCoded = prepareData(dataTraining.copy(), True, XTotalTrain.columns)

# Crear y entrenar el modelo
model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree = 0.4, gamma = 900000.0, learning_rate = 0.3, max_depth = 9, n_estimators = 600, random_state=0)
model.fit(XTotalTrain, yTotalTrain)
# y_pred = model.predict(X_Test)

# Exportar modelo a archivo binario .pkl
joblib.dump(model, 'usedCarPrices.pkl', compress=3) # La ruta antes del nombre del archivo binario debe existir




