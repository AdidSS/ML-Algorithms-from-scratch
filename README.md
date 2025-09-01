# Regresión Logística para Clasificación de Solicitudes de Préstamos Bancarios

![Presentando AIPB](assets/AIPB.png)

## Descripción General del Proyecto

Este proyecto implementa un modelo de regresión logística desde cero para clasificar solicitudes de préstamos bancarios como aprobadas o rechazadas. El objetivo es calsificar las solicitudes calculando la probabilidad de que un préstamo sea aprobado basándose en diversas características de la solicitud, como ingresos del solicitante, puntaje crediticio, monto del préstamo y otros factores relevantes.

## Características

- **Implementación de regresión logística desde cero**: Desarrollo del algoritmo utilizando Python sin depender de frameworks.
- **Preprocesamiento de datos**: Estandarización y técnicas de undersampling para preparar los datos.
- **Entrenamiento y evaluación del modelo**: Optimización de parámetros (pesos y bias) mediante descenso de gradiente.
- **Visualización de resultados**: Curvas ROC, matrices de confusión y gráficos de historial de costos a lo largo de las épocas.

### Preprocesamiento de Datos

- **Estandarización**: Normaliza las características para tener media cero y varianza unitaria, mejorando la convergencia del modelo.
- **Submuestreo (Undersampling)**: Equilibra el conjunto de datos reduciendo la clase mayoritaria, evitando sesgos en las predicciones.
- **División Entrenamiento-Prueba**: Divide los datos en conjuntos de entrenamiento y prueba para una evaluación adecuada del modelo.

### Modelo de Regresión Logística

- **Función Sigmoide**: Transforma predicciones lineales en valores de probabilidad entre 0 y 1.
- **Función de Pérdida LogLoss**: Mide el rendimiento del modelo de clasificación donde la salida es un valor de probabilidad.
- **Descenso de Gradiente**: Algoritmo de optimización que ajusta iterativamente los parámetros del modelo para minimizar la función de costo.

### Métricas de Evaluación

Implementa varias métricas de rendimiento para evaluar el modelo:

| **Métrica**   | **Descripción**                                              |
|---------------|--------------------------------------------------------------|
| Exactitud     | Proporción de instancias correctamente clasificadas          |
| Precisión     | Proporción de identificaciones positivas que fueron correctas|
| Sensibilidad  | Proporción de positivos reales identificados correctamente   |
| Puntuación F1 | Media armónica de precisión y sensibilidad                   |
| Curva ROC     | Gráfico de la tasa de verdaderos positivos frente a la tasa de falsos positivos |
| AUC           | Área bajo la curva ROC, que mide la capacidad de discriminación del modelo |



