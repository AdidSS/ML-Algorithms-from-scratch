import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
#Importar random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#Importar para hacer grid search
from sklearn.model_selection import GridSearchCV
#Para desbalance de datos (over sampling)
from imblearn.over_sampling import RandomOverSampler
#Para estandarizar datos
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
from feature_engine.outliers import Winsorizer
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
import scipy.stats as stats
from scipy.stats import chi2_contingency

## Funciones
def train_val_test_split(X,y, train_size):
  X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, shuffle=True, random_state=42)
  X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size= 0.5, shuffle=True, random_state=42)
  y_train = y_train.values.reshape(len(y_train), 1)
  y_val = y_val.values.reshape(len(y_val), 1)
  y_test = y_test.values.reshape(len(y_test), 1)
  return X_train, y_train, X_val, y_val, X_test, y_test

def sigmoid(x): #Array
    return 1 / (1 + np.exp(-x)) #Array

def log_loss(y, y_predicted): #Array
  N = len(y)
  epsilon = 1e-15
  acum = 0
  for i in range(N):
    if y[i] == 1:
      acum += -np.log(y_predicted[i] + epsilon)
    else:
      acum += -np.log(1 - y_predicted[i] + epsilon)
  return acum / N #Valor del binary cross entrophy

def logistic_regression(X, y, learning_rate, epochs):
  m = X.shape[0]
  n = X.shape[1]
  # Parámetros del modelo: y = x*w + b
  w = np.zeros((n, 1))
  b = 0
  cost_history = []

  for i in range(epochs):
    #Nuestra hipótesis
    y_predicted = sigmoid(np.dot(X, w) + b)
    cost = log_loss(y, y_predicted)

    # Cálculo del gradiente
    # Como estamos haciendo la operacion con matrices no es necesario hacer el
    # ciclo for para iterar los valores de X (m x n)
    dw = (1 / m) * np.dot(X.T, (y_predicted - y))
    db = (1 / m) * np.sum(y_predicted - y)
    # Gradiente descendente para optimizar parámetros
    w = w - learning_rate * dw
    b = b - learning_rate * db
    cost_history.append(cost)
  return w, b, cost_history

def accuracy(y_true, y_pred):
  accuracy = np.sum(y_true == y_pred) / len(y_true)
  return accuracy

def evaluation_report(X_train, X_val, X_test, weights, bias):
  y_pred_train = sigmoid(np.dot(X_train, weights) + bias)
  y_pred_train_class = y_pred_train.round()
  y_pred_val = sigmoid(np.dot(X_val, weights) + bias)
  y_pred_val_class = y_pred_val.round()
  y_predicted = sigmoid(np.dot(X_test, weights) + bias)
  y_predicted_class = y_predicted.round()
  print("TRAIN:")
  print(classification_report(y_train, y_pred_train_class))
  print("VAL:")
  print(classification_report(y_val, y_pred_val_class))
  print("TEST:")
  print(classification_report(y_test, y_predicted_class))

def estadisticas_dataset(X):
  mean = []
  std = []
  for col in X.columns:
    mean.append(np.mean(X[col]))
    std.append(np.std(X[col]))
  return mean, std

def estandarizar(X, mean, std):
  return (X - mean) / std

df = pd.read_csv('loan_data.csv')

# Análisis exploratorio:
print(df.info())
print("Valores nulos: \n" ,df.isna().sum())

#Sin valores nulos, un dataset de 14 columnas y 45,000 registros
df_numerico = df.select_dtypes(include=['int64', 'float64'])
print("Columnas: ",df_numerico.columns)

#pairplot de df numerico
sns.pairplot(df_numerico.sample(1000), hue="loan_status")
plt.show()

num_vars = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
            'credit_score']
fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # 2 filas, 4 columnas
axes = axes.flatten()

for i, col in enumerate(num_vars):
    sns.boxplot(y=df_numerico[col], ax=axes[i])
    axes[i].set_title(col)

plt.tight_layout()
plt.show()

#Correlation matrix as heatmap between variables of df_numerico
corr_matrix = df_numerico.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()

# Variables categóricas:
df_categorico = df.select_dtypes(include=['object'])
df_categorico.columns

#Imprimir los valores únicos de las variables categoricas
for col in df_categorico.columns:
  print(col, df_categorico[col].unique())

#Hacer los plots de la variables categoricas de df, coloreando por loan_status
fig, axes = plt.subplots(2, 3, figsize=(16, 8))  # 2 filas, 4 columnas
axes = axes.flatten()
for i, col in enumerate(df_categorico.columns):
    sns.countplot(x=col, hue="loan_status", data=df, ax=axes[i])
    axes[i].set_title(col)

plt.tight_layout()
plt.show()

categoric_columns = ['person_gender', 'person_education', 'person_home_ownership',
       'loan_intent', 'previous_loan_defaults_on_file']
#Hacer un análisis de correlaciones de variables categoricas respecto a la variable loan_status
#Analizar crosstab y los contrastes de chi cuadrado, su probabilidad p
for col in categoric_columns:
  contigency_table = pd.crosstab(df[col], df['loan_status'])
  print(contigency_table)
  chi2, p, dof, expected = chi2_contingency(contigency_table)
  print("P: ", p)
  if p < 0.05:
    print("Las variables tienen relación significativa")
  else:
    print("Las variables no tienen una relación significativa")
  print("-----")

variables_insignificativas = ['person_gender', 'person_education']

#Dropeamos las columnas que no nos interesan:
df = df.drop(variables_insignificativas, axis=1)

## Tratamiento de valores atípicos:
df = df[df['person_home_ownership'] != 'other']

#Eliminar registros con datos muy atipicos o anormales de acuerdo al contexto de cada columna:
#Eliminar personas con mas de 100 años
df = df[df['person_age'] < 100]
#Eliminar personas con experiencia laboral mayor a 80 años: person_emp_exp
df = df[df['person_emp_exp'] < 80]
#personas con un income mucho mayor a 5*1*e6
df = df[df['person_income'] < 5e6]

var = ['person_emp_exp','cb_person_cred_hist_length']
df = df.drop(var, axis=1)

num_vars = ['person_age', 'person_income', 'loan_amnt',
            'loan_int_rate', 'loan_percent_income',
            'credit_score']

print(df[df["loan_percent_income"] == 0].count())
df = df[df["loan_percent_income"] != 0]

df_numerico = df[['person_age', 'person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'credit_score']]
mean, std = estadisticas_dataset(df_numerico)
df_numerico_estandarizado = estandarizar(df_numerico, mean, std)
#Reemplazar valores en df
df[df_numerico.columns] = df_numerico_estandarizado
df.head()

#visualización de las distribuciones de df_numerico
plt.figure(figsize=(16, 8))  # Crea una nueva figura con tamaño específico
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(df_numerico_estandarizado.columns):
    sns.histplot(df_numerico_estandarizado[col], ax=axes[i])
    axes[i].set_title(col)
plt.tight_layout()
plt.show()

#windsorize numeric columns of df ['person_age', 'person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'credit_score']
windsorizer = Winsorizer(capping_method='gaussian', tail='both', fold=3, variables=['person_age', 'person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'credit_score'])

df_windsorized = windsorizer.fit_transform(df)

plt.figure(figsize=(16, 8))  # Crea una nueva figura con tamaño específico
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(df_numerico_estandarizado.columns):
    sns.histplot(df_windsorized[col], ax=axes[i])
    axes[i].set_title(col)
plt.tight_layout()
plt.show()

#Encoding variables categoricas:
#Categoric columns ['person_home_ownership','loan_intent','previous_loan_defaults_on_file']
#One hot encoding:
df_windsorized = pd.get_dummies(df_windsorized, columns=['person_home_ownership','loan_intent','previous_loan_defaults_on_file'])
#Cambiar dummies por 0 o 1
for col in df_windsorized.columns:
  if True in df_windsorized[col].unique():
    df_windsorized[col] = df_windsorized[col].apply(lambda x: 1 if x == True else 0)

#Dropeamos ya sea previous_loan_defaults_on_file_Yes o la otra, ya que son contrarias
df_windsorized = df_windsorized.drop(['previous_loan_defaults_on_file_Yes'], axis=1)
print("Columnas después del encoding: ")
print(df_windsorized.columns)

# Desbalanceo de los datos
print(df_windsorized['loan_status'].value_counts())
plt.figure(figsize=(8, 6))  # Crea una nueva figura
sns.countplot(x='loan_status', data=df_windsorized)
plt.title('Distribución de loan_status')
plt.show()

#Oversampling con SMOTE
smote = SMOTE(random_state=42)
X = df_windsorized.drop('loan_status', axis=1)
y = df_windsorized['loan_status']
X_SMOTE, y_SMOTE = smote.fit_resample(X, y)

#COntar registros
print(y_SMOTE.value_counts())

#Oversampling duplicando los registros de la clase minoritaria
ros = RandomOverSampler(random_state=42)
X_Over, y_Over = ros.fit_resample(X, y)

print(y_Over.value_counts())

#PODEMOS CAMBIAR A SMOTE u OVER
X = df_windsorized[['person_age', 'person_income', 'loan_amnt', 'loan_int_rate',
       'loan_percent_income', 'credit_score',
       'person_home_ownership_MORTGAGE',
       'person_home_ownership_OWN', 'person_home_ownership_RENT',
       'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION',
       'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
       'loan_intent_PERSONAL', 'loan_intent_VENTURE',
       'previous_loan_defaults_on_file_No']]

y = df_windsorized['loan_status']

# Entrenamiento Con desbalance:
print("Logistic Regression sin balancear \n")
X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y, 0.8)
weights, bias, cost_history = logistic_regression(X_train, y_train, 0.1, 200)
print(weights)
print(bias)
evaluation_report(X_train, X_val, X_test, weights, bias)

#Con SMOTE
#Contar tiempo de entrenamiento
start_time = time.time()
print("Logistic Regression con SMOTE \n")
X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X_SMOTE, y_SMOTE, 0.8)
weights, bias, cost_history = logistic_regression(X_train, y_train, 0.1, 200)
end_time = time.time()
print("Tiempo de entrenamiento: ", end_time - start_time, " segundos")
#Hacer las predicciones para el x_test
y_predicted = sigmoid(np.dot(X_test, weights) + bias)
#evaluar predicciones:
y_predicted_class = y_predicted.round()
confusion_matrix(y_test, y_predicted_class)
sns.heatmap(confusion_matrix(y_test, y_predicted_class), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
evaluation_report(X_train, X_val, X_test, weights, bias)
# Grafica del cost history a traves de las epocas:
plt.plot(cost_history)
plt.xlabel("Epocas")
plt.ylabel("Cost")
plt.title("Cost vs Epocas")
plt.show()

#Con over sampling
print("Logistic Regression con Over Sampling \n")
X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X_Over, y_Over, 0.8)
weights, bias, cost_history = logistic_regression(X_train, y_train, 0.1, 200)
evaluation_report(X_train, X_val, X_test, weights, bias)

# Implementación de un ensemble method con framework
start_time = time.time()
print("Random Forest Classifier con SMOTE \n")
X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X_SMOTE, y_SMOTE, 0.8)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
end_time = time.time()
print("Tiempo de entrenamiento: ", end_time - start_time, " segundos")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
#Plot the ROC curve for the train and test model
y_train_pred = model.predict_proba(X_train)[:, 1]
y_test_pred = model.predict_proba(X_test)[:, 1]
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred)
plt.plot(fpr_train, tpr_train, label='Train')
plt.plot(fpr_test, tpr_test, label='Test')
#Grafica del valor de cada coeficiente del modelo
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X_train.columns
plt.figure(figsize=(10, 6))
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), feature_names[indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()
print(importances)

#Evauación en los splits
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)
print("TRAIN:")
print(classification_report(y_train, y_train_pred))
print("VAL:")
print(classification_report(y_val, y_val_pred))
print("TEST:")
print(classification_report(y_test, y_test_pred))
#Obtener los hiperparametros del modelo
print("Hiperparámetros del modelo Random Forest:")
print(model.get_params())


"""
#Optimización de hiperparámetros con Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
print("Best parameters found: ", grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
print(evaluation_report(X_train, X_val, X_test, weights, bias))
#Plot the ROC curve for the train and test model
y_train_pred = model.predict_proba(X_train)[:, 1]
y_test_pred = model.predict_proba(X_test)[:, 1]
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred)
plt.plot(fpr_train, tpr_train, label='Train')
plt.plot(fpr_test, tpr_test, label='Test')
#Best parameters found:  {'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
"""