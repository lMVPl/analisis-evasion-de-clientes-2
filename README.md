# 📈 Predicción de Cancelación de Clientes (Churn)

Este proyecto tiene como objetivo desarrollar modelos predictivos capaces de **identificar qué clientes tienen mayor probabilidad de cancelar sus servicios**, basándose en variables relacionadas al cliente, al contrato y al uso de servicios.

---

## 🎯 Propósito del Análisis

El objetivo principal fue aplicar técnicas de análisis de datos y machine learning para:

- **Comprender los factores que influyen en la cancelación de clientes (churn)**.
- **Desarrollar modelos predictivos** para anticipar qué clientes están en riesgo.
- **Proponer estrategias de retención** basadas en los resultados obtenidos.

---

## 🗂️ Estructura del Proyecto

```bash
├── churn_analysis.ipynb # Cuaderno principal con todo el análisis
├── datos-limpios_evasion-clientes.csv # Dataset limpio y preprocesado
├── visualizaciones/ # Carpeta opcional para guardar gráficos
└── README.md # Descripción del proyecto
```

---

## 🧹 Preparación y Tratamiento de Datos

### 1. Clasificación de variables

- **Numéricas continuas**:  
  `tenure`, `MonthlyCharges`, `TotalCharges`, `cuentas_diarias`
  
- **Categóricas** (binarizadas mediante one-hot encoding):  
  `gender`, `Partner`, `Dependents`, `InternetService`, `PaymentMethod`, `Contract`, `PaperlessBilling`, `TechSupport`, entre otras.

### 2. Limpieza y transformación

- Se eliminaron columnas sin valor predictivo como `customerID`.
- Se realizó **codificación One-Hot** para variables categóricas.
- Algunas variables fueron normalizadas con `StandardScaler` (solo para modelos que lo requieren).

### 3. División de los datos

- Se separaron los datos en **conjuntos de entrenamiento (80%) y prueba (20%)**, estratificados por clase (`Churn`).
- Se aplicó **SMOTE** para balancear las clases en el conjunto de entrenamiento, debido al desbalance original (73% clase 0, 27% clase 1).

---

## 🤖 Modelos Utilizados

Se entrenaron dos modelos principales:

| Modelo              | Requiere Normalización | Accuracy | Precision (Churn) | Recall (Churn) |
|---------------------|------------------------|----------|-------------------|----------------|
| Regresión Logística | ✅ Sí                  | 0.79     | 0.63              | 0.52           |
| Árbol de Decisión   | ❌ No                  | 0.72     | 0.47              | 0.47           |

La **Regresión Logística** tuvo mejor desempeño global en métricas clave para detectar churn.

---

## 📊 Análisis Exploratorio (EDA)

Se realizaron varios análisis visuales y estadísticos:

### 🔍 Ejemplos de insights:

- Clientes con **contratos mensuales**, **poca antigüedad** (`tenure`) y que pagan por **cheque electrónico** tienen más riesgo de cancelar.
- Contratos de **dos años** y clientes con servicios como **seguridad en línea** o **soporte técnico** tienden a quedarse.

### 📈 Ejemplos de gráficos:

- `barplot` de proporción de cancelación por tipo de contrato. 
- Matriz de correlación entre variables numéricas y `Churn`.
- Importancia de variables según cada modelo.

---

## 🛠️ Instrucciones para ejecutar el proyecto

1. Abre el archivo `churn_analysis.ipynb` en Google Colab o Jupyter.
2. Asegúrate de instalar las siguientes bibliotecas si no están preinstaladas:

```bash
pip install pandas numpy scikit-learn seaborn imbalanced-learn matplotlib
