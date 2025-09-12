# Modelos de Machine Learning para Clasificación de Solicitudes de Préstamos Bancarios

![Presentando AIPB](assets/AIPB.png)

## Descripción General del Proyecto

Este proyecto implementa un modelo de machine learning (regresión logística) desde cero y un random forest con scikit learn, para clasificar solicitudes de préstamos bancarios como aprobadas o rechazadas. El objetivo es clasificar las solicitudes calculando la probabilidad de que un préstamo sea aprobado basándose en diversas características de la solicitud, como ingresos del solicitante, puntaje crediticio, monto del préstamo y otros factores relevantes.

## Estructura del Repositorio


| **Archivo**   | **Descripción**                                              |
|---------------|--------------------------------------------------------------|
| loan_data.csv     | Dataset utilizado para el proyecto          |
| Machine_Learning_Model_From_Scratch.pdf     | Avance del proyecto|
| main.py  | Código con la implementación del modelo de ML sin frameworks   |
| aprobacion_credito_ml.py  | Código final (implementación con y sin framework) |
| Loan Approval with Machine Learning.pdf   | Reporte Final |

## Instalación y Ejecución

Para ejecutar este proyecto en tu entorno local, sigue estos pasos:

### Prerrequisitos

- pip (gestor de paquetes de Python)
- Git (opcional, para clonar el repositorio)

### Configuración del entorno

1. Clona el repositorio (o descárgalo como ZIP):
   ```bash
   git clone https://github.com/tu-usuario/ML-Algorithms-from-scratch.git
   cd ML-Algorithms-from-scratch
   ```
2. Crea un entorno virtual:
    En Windows:

   ```bash
    python -m venv venv
    venv\Scripts\activate
   ```

    En macOS/Linux:
   ```bash
    python3 -m venv venv
    source venv/bin/activate
   ```

3. Instala las dependencias:
   ```bash
    pip install -r requirements.txt
   ```

#### Ejecución:
Primer modelo de regresión logística:

```bash
python main.py
```
Para ejecutar el análisis completo con implementación desde cero y Random Forest:
```bash
python aprobacion_credito_ml.py
```

