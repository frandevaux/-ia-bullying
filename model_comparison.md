# Random Forest vs Support Vector Machine

Se utilizaron los algoritmos de Random Forest y Support Vector Machine para predecir la columna 'Bullied_in_past_12_months' del dataset, clasificando a los estudiantes en dos clases: 'Bullied' si sufrieron bullying en los últimos 12 meses y 'Not bullied', en caso contrario.

'Bullied_in_past_12_months' se crea a partir de combinar otras 3 features referidas al bullying: Bullied_on_school_property_in_last_12_months, Bullied_not_on_school_property_in_last_12_months y Cyber_Bullied_in_last_12_months; si alguno de los 3 es true Bullied_in_last_12_months es true.

En cada enfoque se optó por utilizar distintas features debido a que mejoraban el rendimiento general de cada uno, de acuerdo a las métricas elegidas.

# Random Forest

A partir de una implementación de random forest con las siguientes features:

- Bullied_in_past_12_months
- Physically_attacked
- Physical_fighting
- Felt_lonely

Y prediciendo la columna 'Bullied_in_past_12_months', generando 250 arboles, se obtuvieron los siguientes resultados:

## Train

### Matriz de confusión

|              | Predicted 0 | Predicted 1 |
| ------------ | ----------- | ----------- |
| **Actual Not bullied** | 19979       | 4683        |
| **Actual Bullied** | 8822        | 7404        |

### Reporte de la clasificación

|                  | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| **Not bullied**      | 0.69      | 0.81   | 0.75     | 24662   |
| **Bullied**      | 0.61      | 0.46   | 0.52     | 16226   |
| **Accuracy**     |           |        | 0.67     | 40888   |
| **Macro Avg**    | 0.65      | 0.63   | 0.64     | 40888   |
| **Weighted Avg** | 0.66      | 0.67   | 0.66     | 40888   |

## Test

### Matriz de confusión

|              | Predicted 0 | Predicted 1 |
| ------------ | ----------- | ----------- |
| **Actual Not bullied** | 4927        | 1171        |
| **Actual Bullied** | 2204        | 1920        |

### Reporte de la clasificación

|                  | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| **Not bullied**      | 0.69      | 0.81   | 0.74     | 6098    |
| **Bullied**      | 0.62      | 0.47   | 0.53     | 4124    |
| **Accuracy**     |           |        | 0.6710   | 10222   |
| **Macro Avg**    | 0.66      | 0.64   | 0.64     | 10222   |
| **Weighted Avg** | 0.66      | 0.67   | 0.66     | 10222   |

### Gráficos

Generamos gráficos con el propósito de identificar la combinación óptima de la cantidad de árboles a utilizar en el modelo Random Forest, junto con los mejores pesos para las clases respectivas. Al analizar los resultados, observamos que no hubo una mejora significativa al aumentar el número de árboles, por lo que decidimos mantener n=100. En cuanto a los pesos de las clases, optamos por una opción equilibrada entre el accuracy y el recall, seleccionando Not_bullied: 1 y Bullied: 1.5 (representados por la curva verde).

![rf_grid_search_accuracy.png](./results/plots/rf_grid_search_accuracy.png)

![rf_grid_search_recall.png](./results/plots/rf_grid_search_recall.png)

# Support Vector Machine

Para la implementación de este algoritmo se utilizaron los siguientes parámetros:

- kernel='rbf': Se utiliza un kernel radial, que es comúnmente utilizado en problemas no lineales.

- C=10: El parámetro C controla la penalización por error en la clasificación. Un valor más alto de C hará que el modelo sea más estricto, tratando de clasificar correctamente todos los puntos de entrenamiento, pero puede llevar a overfitting.

- gamma= 0.001: El parámetro gamma controla la amplitud de la función kernel. Un valor bajo de gamma produce una función kernel más suave, mientras que un valor alto puede llevar a overfitting.

- probability=True: Este parámetro habilita el cálculo de probabilidades de pertenencia a cada clase.
  
- class_weight= {0: 1, 1: 1.5}

## Train

### Matriz de confusión

|              | Predicted 0 | Predicted 1 |
| ------------ | ----------- | ----------- |
| **Actual Not bullied** | 18978       | 5684        |
| **Actual Bullied** | 8661        | 7565        |

### Reporte de la clasificación

|                  | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| **Not bullied**      | 0.69      | 0.77   | 0.73     | 24662   |
| **Bullied**      | 0.57      | 0.47   | 0.51     | 16226   |
| **Accuracy**     |           |        | 0.65     | 40888   |
| **Macro Avg**    | 0.63      | 0.62   | 0.62     | 40888   |
| **Weighted Avg** | 0.64      | 0.65   | 0.64     | 40888   |

## Test

### Matriz de confusión

|              | Predicted 0 | Predicted 1 |
| ------------ | ----------- | ----------- |
| **Actual Not bullied** | 4767        | 1331        |
| **Actual Bullied** | 2192        | 1932        |

### Reporte de la clasificación

|                  | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| **Not bullied**      | 0.69      | 0.78   | 0.73     | 6098    |
| **Bullied**      | 0.59      | 0.47   | 0.52     | 4124    |
| **Accuracy**     |           |        | 0.66     | 10222   |
| **Macro Avg**    | 0.64      | 0.63   | 0.63     | 10222   |
| **Weighted Avg** | 0.65      | 0.66   | 0.65     | 10222   |

### Gráficos

Se generaron gráficos que representan el accuracy y el recall en función del factor de penalización de SVM, utilizando 15 divisiones diferentes. A partir de estos resultados, se seleccionó el valor de c=10, ya que demostró ser la elección que logra el mejor equilibrio entre ambas métricas.

![svm_boxplot_c_accuracy.png](./results/plots/svm_boxplot_c_accuracy.png)

![svm_boxplot_c_recall.png](./results/plots/svm_boxplot_c_recall.png)


# Comparación de métricas de los modelos

![boxplot_rf_metrics](./results/plots/rf_boxplot_30.png)

![boxplot_svm_metrics](./results/plots/svm_boxplot_30.png)




