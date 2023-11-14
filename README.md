# Detección de casos de bullying

### Código del proyecto: BULL

### Integrantes: Francisco Devaux y Bautista Frigolé

### Descripción

Nuestro proyecto de inteligencia artificial se enfoca en la detección y predicción de casos de bullying utilizando técnicas de machine learning. El objetivo principal de esta iniciativa es desarrollar un modelo capaz de identificar situaciones de acoso entre estudiantes, con el fin de prevenir y abordar este grave problema que afecta a jóvenes en todo el mundo. Para lograr esto, hemos empleado el [dataset](https://www.kaggle.com/datasets/leomartinelli/bullying-in-schools) del Global School-Based Student Health Survey (GSHS) realizado en Argentina en 2018, que proporciona una amplia gama de información relevante sobre la salud y el comportamiento de los jóvenes.

El Global School-Based Student Health Survey (GSHS) es una encuesta basada en escuelas que utiliza un cuestionario autoadministrado para recopilar datos sobre el comportamiento de salud de los jóvenes y los factores protectores relacionados con las principales causas de morbilidad y mortalidad. En la edición realizada en Argentina en 2018, participaron un total de 56,981 estudiantes.

El dataset cuenta con 18 variables entre las cuales se destacan:

- Suffered bullying in past 12 months
- Age
- Sex
- Felt lonely
- Close friends amount
- Miss school with no permission
- Parents understand problems
- Were underweight
- Were obese

### Objetivo

El objetivo fundamental de nuestro proyecto es desarrollar un modelo de machine learning que permita predecir situaciones de bullying en base a las demás variables. Al utilizar técnicas avanzadas de análisis de datos y aprendizaje automático, aspiramos a identificar patrones y relaciones ocultas en los datos que nos permitan anticipar casos de bullying, proporcionando así una herramienta efectiva para la prevención y el apoyo a los estudiantes afectados.

### Justificación

Al desarrollar un modelo de predicción sólido, se puede brindar una herramienta valiosa para la detección temprana y la prevención del bullying mediante encuestas en grandes cantidades de estudiantes, algo que no podría ser posible sin el uso de técnicas de inteligencia artificial, avisando de estos posibles casos de bullying a profesionales para que tomen cartas en el asunto. Este modelo podría contribuir al bienestar de los estudiantes y a la creación de un entorno escolar más seguro y saludable.

### Métricas

Para evaluar la eficacia del modelo de detección de bullying, utilizaremos un conjunto de métricas clásicas en problemas de clasificación. Estas métricas proporcionarán una visión más completa del rendimiento del modelo, complementando la tasa de observaciones correctamente detectadas. Aquí están las métricas que consideraremos:

1. **Precisión (Precision):**
   - Descripción: La precisión mide la proporción de instancias positivas correctamente clasificadas respecto a todas las instancias clasificadas como positivas.
   - Fórmula: Precision = TP / (TP + FP)
   - Interpretación: Proporción de casos positivos predichos correctamente entre todos los casos predichos como positivos.

2. **Recall (Sensibilidad o Tasa de Verdaderos Positivos):**
   - Descripción: El recall mide la proporción de instancias positivas correctamente clasificadas respecto a todas las instancias que son realmente positivas.
   - Fórmula: Recall = TP / (TP + FN)
   - Interpretación: Proporción de casos positivos predichos correctamente entre todos los casos reales positivos.

3. **F1-Score:**
   - Descripción: El F1-score es la media armónica de la precisión y el recall, proporcionando un equilibrio entre ambas métricas.
   - Fórmula: F1 = 2 * (Precision * Recall) / (Precision + Recall)
   - Interpretación: Combina precisión y recall en una única métrica, ideal para problemas con desequilibrio de clases.

4. **Exactitud (Accuracy):**
   - Descripción: La exactitud mide la proporción de instancias correctamente clasificadas, independientemente de la clase.
   - Fórmula: Accuracy = (TP + TN) / (TP + TN + FP + FN)
   - Interpretación: Proporción de todas las predicciones correctas.

5. **Especificidad (Tasa de Verdaderos Negativos):**
   - Descripción: La especificidad mide la proporción de instancias negativas correctamente clasificadas respecto a todas las instancias realmente negativas.
   - Fórmula: Especificidad = TN / (TN + FP)
   - Interpretación: Proporción de casos negativos predichos correctamente entre todos los casos reales negativos.

6. **Curva ROC y Área bajo la Curva (AUC-ROC):**
   - Descripción: La curva ROC representa la tasa de verdaderos positivos frente a la tasa de falsos positivos para diferentes umbrales de decisión. El AUC-ROC mide la capacidad del modelo para distinguir entre clases.
   - Interpretación: Un AUC-ROC cercano a 1 indica un buen rendimiento del modelo.

Es importante evaluar estas métricas en conjunto, ya que proporcionan información valiosa sobre diferentes aspectos del rendimiento del modelo. Dependiendo de los requisitos específicos y las implicaciones prácticas del problema, algunas métricas pueden tener más relevancia que otras.
