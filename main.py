import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
start_time = time.time()

#Estandarization:
def estadisticas_dataset(X):
  mean = []
  std = []
  for col in X.columns:
    mean.append(np.mean(X[col]))
    std.append(np.std(X[col]))
  return mean, std

def estandarizar(X, mean, std):
  return (X - mean) / std

#Funcion de undersampling para tener la misma cantidad de statu loan 0 y status loan 1
def undersampling(X, y):
  min_class = np.min(y.value_counts())
  indices_0 = y[y == 0].index
  indices_1 = y[y == 1].index
  indices_0 = np.random.choice(indices_0, size=min_class, replace=False)
  indices_1 = np.random.choice(indices_1, size=min_class, replace=False)
  indices = np.concatenate([indices_0, indices_1])
  X_undersampled = X.loc[indices]
  y_undersampled = y.loc[indices]
  # Resetear indices:
  X_undersampled = X_undersampled.reset_index(drop=True)
  y_undersampled = y_undersampled.reset_index(drop=True)
  return X_undersampled, y_undersampled

def train_test_split(X, y, train_size):
    if train_size > 1 or train_size <= 0:
        raise ValueError("Agarra un train size válido")

    n = len(X)
    train_size = int(train_size * n)

    # Shufflear dataset
    indices = np.random.permutation(n)
    X = X.iloc[indices]
    y = y.iloc[indices]

    # Split
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]

    return X_train, y_train, X_test, y_test
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
  train_start_time = time.time()
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

  train_time = time.time() - train_start_time
  print(f"\nTiempo de entrenamiento: {train_time:.2f} segundos")
  return w, b, cost_history

#Evaluacion del modelo:
def confusion_matrix(y_true, y_pred):
  tp = np.sum((y_true == 1) & (y_pred == 1))
  tn = np.sum((y_true == 0) & (y_pred == 0))
  fp = np.sum((y_true == 0) & (y_pred == 1))
  fn = np.sum((y_true == 1) & (y_pred == 0))
  return tp, tn, fp, fn

def calculate_metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
def plot_confusion_matrix(y_true, y_pred, labels=['Rechazado', 'Aprobado']):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    cm = np.array([[tn, fp], 
                   [fn, tp]])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels,
                yticklabels=labels)
    plt.title('Matriz de Confusión')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predicho')
    plt.show()

def plot_roc_curve(y_true, y_pred_proba):
    """
    Grafica la curva ROC y calcula el AUC.
    
    Args:
        y_true: Valores reales
        y_pred_proba: Probabilidades predichas
    """
    # Calcular TPR y FPR para diferentes umbrales
    thresholds = np.linspace(0, 1, 100)
    tpr_list = []
    fpr_list = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
        tpr = tp / (tp + fn) if (tp + fn) != 0 else 0  # Sensitivity
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0  # 1 - Specificity
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    # Calcular AUC usando la regla del trapecio
    auc = np.trapz(tpr_list, fpr_list)
    
    # Graficar
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_list, tpr_list, 'b-', label=f'ROC (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Random')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Curva ROC')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_count_classes(df, column):
   sns.countplot(x=column, data=df)
   plt.show()

#Ejecución:
df = pd.read_csv('loan_data.csv')
print(df.info())
print("Valores nulos",df.isna().sum())
df_numerico = df.select_dtypes(include=['int64', 'float64'])
print("Columnas: ",df_numerico.columns)
sns.pairplot(df_numerico.sample(2000), hue = "loan_status")
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
#Eliminar registros con datos muy atipicos o anormales de acuerdo al contexto de cada columna:
#Eliminar personas con mas de 100 años
df_numerico = df_numerico[df_numerico['person_age'] < 100]
#Eliminar personas con experiencia laboral mayor a 80 años: person_emp_exp
df_numerico = df_numerico[df_numerico['person_emp_exp'] < 80]
#personas con un income mucho mayor a 5*1*e6
df_numerico = df_numerico[df_numerico['person_income'] < 5e6]
#Correlation matrix as heatmap between variables of df_numerico
corr_matrix = df_numerico.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()
X = df_numerico[['person_age', 'person_income', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income',
       'credit_score']]

y = df_numerico['loan_status']
#Cuantas clases hay:
plot_count_classes(df_numerico, 'loan_status')

#Probar regresion logistica con df_numerico
X, y = undersampling(X, y)
mean, std = estadisticas_dataset(X)
X_estandarizado = estandarizar(X, mean, std)
print(X_estandarizado.head())
print(len(X_estandarizado))
X_train, y_train, X_test, y_test = train_test_split(X_estandarizado, y, 0.8)
y_train = y_train.values.reshape(len(y_train), 1)
y_test = y_test.values.reshape(len(y_test), 1)
weights, bias, cost_history = logistic_regression(X_train, y_train, 0.1, 100)
print(y_train.shape)
print("Pesos:", weights)
print("Bias:", bias)

# Grafica del cost history a traves de las epocas:
plt.plot(cost_history)
plt.xlabel("Epocas")
plt.ylabel("Cost")
plt.title("Cost vs Epocas")
plt.show()
#Hacer las predicciones para el x_test
y_predicted = sigmoid(np.dot(X_test, weights) + bias)
#evaluar predicciones:
y_predicted_class = y_predicted.round()

#Evaluación:
plot_roc_curve(y_test, y_predicted)
plot_confusion_matrix(y_test, y_predicted_class)
metrics = calculate_metrics(y_test, y_predicted_class)
print("Accuracy:", metrics['accuracy'])
print("Precision:", metrics['precision'])
print("Recall:", metrics['recall'])
print("F1 Score:", metrics['f1_score'])
total_time = time.time() - start_time
print(f"\nTiempo total de ejecución: {total_time:.2f} segundos")