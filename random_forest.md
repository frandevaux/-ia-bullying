# Random Forest

Estos son los resultados obtenidos con el modelo de random forest:

## Matriz de confusión

|              | Predicted 0 | Predicted 1 |
| ------------ | ----------- | ----------- |
| **Actual 0** | 5156        | 1660        |
| **Actual 1** | 2414        | 2113        |

## Reporte de la clasificación

|                  | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| **Class 0**      | 0.68      | 0.76   | 0.72     | 6816    |
| **Class 1**      | 0.56      | 0.47   | 0.51     | 4527    |
| **Accuracy**     |           |        | 0.6408   | 11343   |
| **Macro Avg**    | 0.62      | 0.61   | 0.61     | 11343   |
| **Weighted Avg** | 0.63      | 0.64   | 0.63     | 11343   |

A partir de la nueva implementación tenemos los siguientes resultados:

## Matriz de confusión

|              | Predicted 0 | Predicted 1 |
| ------------ | ----------- | ----------- |
| **Actual 0** | 5119        | 95          |
| **Actual 1** | 1283        | 91          |

## Reporte de la clasificación

|                  | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| **Class 0**      | 0.80      | 0.98   | 0.88     | 5214    |
| **Class 1**      | 0.49      | 0.07   | 0.12     | 1374    |
| **Accuracy**     |           |        | 0.7908   | 6588    |
| **Macro Avg**    | 0.64      | 0.52   | 0.50     | 6588    |
| **Weighted Avg** | 0.73      | 0.79   | 0.72     | 6588    |
