# Detección de casos de bullying

**Código del proyecto: BULL**

**Integrantes: Francisco Devaux (13848) y Bautista Frigolé (13675)**

## Introducción

En el presente informe se aborda un problema de *inteligencia artificial*, específicamente en el campo del aprendizaje automático y ciencia de datos. Nuestro objetivo es probar dos algoritmos (Random Forest y Support Vector Machine) y crear un *modelo de clasificación*, para que dado un dataset, éste sea capaz de identificar situaciones de bullying entre estudiantes, con el fin de prevenir y abordar este grave problema que afecta a los jóvenes.

La elección de estos algoritmos responde a la necesidad de modelar relaciones no lineales, gestionar conjuntos de datos extensos y enfrentar posibles desequilibrios en la distribución de clases. Estos métodos destacan por su capacidad para manejar múltiples características y proporcionar resultados robustos, lo cual es esencial en un entorno tan diverso como el estudiantil.

No obstante, los modelos deben ser cuidadosamente ajustados y evaluados para garantizar resultados precisos y relevantes. Además, la interpretación de los resultados puede requerir consideraciones éticas y contextualización adecuada dada la naturaleza sensible de los datos de salud estudiantil.

A lo largo de este informe iremos explicando un marco teórico de cada uno de los algoritmos mencionados, el experimento práctico con la configuración y los resultados de los modelos, y las conclusiones resultantes, realizando una comparación de ambos modelos, teniendo en cuenta sus ventajas y desventajas.

## Marco teórico

### Random Forest


**Árboles de decisión**

El algoritmo de Random Forest se compone de la construcción de múltiples arboles de decisión, concepto el cual se explicará brevemente a continuación.
Los árboles de decisión buscan contestar una pregunta de sí o no, como por ejemplo  "¿Debería navegar?" A partir de ahí, puede hacer una serie de preguntas para determinar una respuesta, como, "¿Es un oleaje prolongado?" o "¿El viento sopla en alta mar?". 
Estas son las preguntas que construirán los nodos de decisión del árbol, cuyas respuestas van a dirigir a una nueva pregunta hasta llegar a un nodo hoja que de una respuesta de sí o no.  
Este árbol de decisiones es un ejemplo de un problema de clasificación, donde las etiquetas de clase son "navegar" y "no navegar".
Si bien los árboles de decisión son algoritmos comunes de aprendizaje supervisado, pueden ser propensos a problemas, como sesgos y sobreajuste. Sin embargo, cuando varios árboles de decisión forman un conjunto en el algoritmo de random forest, predicen resultados más precisos, especialmente cuando los árboles individuales no están correlacionados entre sí.

**Métodos de conjunto**

Otro concepto importante para entender el funcionamiento de Random Forest es el de los métodos de aprendizaje por conjuntos, los cuales se componen de un conjunto de clasificadores, por ejemplo, árboles de decisión, y sus predicciones se agregan para identificar el resultado más popular.
Los métodos de conjunto más conocidos son el ensacado, también conocido como agregación de arranque, y el impulso. En el primer método, se selecciona una muestra aleatoria de datos en un conjunto de entrenamiento con reemplazo, lo que significa que los puntos de datos individuales se pueden elegir más de una vez. Después de generar varias muestras de datos, estos modelos se entrenan de forma independiente y, según el tipo de tarea, es decir, regresión o clasificación, el promedio o la mayoría de esas predicciones arrojan una estimación más precisa. Este enfoque se usa comúnmente para reducir la variación dentro de un conjunto de datos ruidoso.

**Algoritmo de Random Forest**

El algoritmo de Random Forest es una extensión del método de ensacado, ya que utiliza tanto el ensacado como la aleatoriedad de características para crear un bosque no correlacionado de árboles de decisión.

La aleatoriedad de features, también conocida como agrupación de características o "el método del subespacio aleatorio ", genera un subconjunto aleatorio de características, lo que garantiza una baja correlación entre los árboles de decisión. Ésta es una diferencia clave entre los árboles de decisión y los bosques aleatorios.

Mientras que los árboles de decisión consideran todas las posibles divisiones de características, los bosques aleatorios solo seleccionan un subconjunto de esas características.

Si volvemos a la pregunta "¿debería navegar?" Por ejemplo, las preguntas que puedo hacer para determinar la predicción pueden no ser tan completas como el conjunto de preguntas de otra persona. Al tener en cuenta toda la variabilidad potencial en los datos, podemos reducir el riesgo de sobreajuste, sesgo y varianza general, lo que da como resultado predicciones más precisas.

**¿Cómo funciona los algoritmos de random forest?**

Los algoritmos de random forest tienen tres hiperparámetros principales, que deben configurarse antes del entrenamiento:

- Tamaño del nodo
- Cantidad de árboles
- Cantidad de características muestreadas

A partir de ahí, el clasificador de random forest se puede utilizar para solucionar problemas de regresión o clasificación.

El algoritmo de random forest se compone de un conjunto de árboles de decisión, y cada árbol del conjunto se compone de una muestra de datos extraída de un conjunto de entrenamiento con reemplazo, llamada muestra de arranque.

De esa muestra de entrenamiento, un tercio se reserva como datos de prueba, lo que se conoce como muestra fuera de la bolsa (oob), a la que volveremos más adelante. Luego, se inyecta otra instancia de aleatoriedad a través del agrupamiento de características, lo que agrega más diversidad al conjunto de datos y reduce la correlación entre los árboles de decisión.

Dependiendo del tipo de problema, la determinación de la predicción variará. Para una tarea de regresión, se promediarán los árboles de decisión individuales, y para una tarea de clasificación, un voto mayoritario, es decir, la variable categórica más frecuente, arrojará la clase predicha.

Finalmente, la muestra de oob se utiliza para la validación cruzada, finalizando esa predicción.

**Ventajas y desafíos del bosque aleatorio**

Hay una serie de ventajas y desafíos clave que presenta el algoritmo de random forest cuando se usa para problemas de clasificación o regresión:
Beneficios clave
 
Riesgo reducido de sobreajuste

Los árboles de decisión corren el riesgo de sobreajustarse, ya que tienden a ajustar todas las muestras dentro de los datos de entrenamiento. Sin embargo, cuando hay una gran cantidad de árboles de decisión en un random forest, el clasificador no se ajustará demasiado al modelo, ya que el promedio de árboles no correlacionados reduce la varianza general y el error de predicción.
Aporta flexibilidad

Dado que el random forest puede manejar tareas de regresión y clasificación con un alto grado de precisión, es un método popular entre los científicos de datos. El agrupamiento de características también convierte al clasificador de random forest en una herramienta eficaz para estimar los valores perdidos, ya que mantiene la precisión cuando falta una parte de los datos.

Importancia de la característica fácil de determinar

El random forest facilita la evaluación de la importancia o contribución de las variables al modelo. Hay algunas formas de evaluar la importancia de las características. La importancia de Gini y la disminución media de impurezas (MDI) se utilizan generalmente para medir cuánto disminuye la precisión del modelo cuando se excluye una variable determinada.

Sin embargo, la importancia de la permutación, también conocida como precisión de disminución media (MDA), es otra medida de importancia. MDA identifica la disminución promedio en la precisión mediante la permutación aleatoria de los valores de las características en las muestras oob.


Explicación

Ventajas y Desventajas

Justificación

### Support Vector Machine

Explicación

Ventajas y Desventajas

Justificación

## Diseño Experimental

### Dataset

Hemos empleado el [dataset](https://www.kaggle.com/datasets/leomartinelli/bullying-in-schools) del Global School-Based Student Health Survey (GSHS) realizado en Argentina en 2018. El Global School-Based Student Health Survey (GSHS) es una encuesta basada en escuelas que utiliza un cuestionario autoadministrado para recopilar datos sobre el comportamiento de salud de los jóvenes y los factores protectores relacionados con las principales causas de morbilidad y mortalidad. En la edición realizada en Argentina en 2018, participaron un total de 56,981 estudiantes.

**Features:**

- Bullied on school property in past 12 months
- Bullied not on school property in past 12_months
- Cyber bullied in past 12 months
- Custom Age
- Sex
- Physically attacked
- Physical fighting
- Felt lonely
- Close friends
- Miss school no permission
- Other students kind and helpful
- Parents understand problems
- Most of the time or always felt lonely
- Missed classes or school without permission
- Were underweight
- Were overweight
- Were obese

### Preprocesamiento y análisis exploratorio de datos

Se decide crear una única feature referida al bullying: 'Bullied_in_past_12_months', la cual es la combinación de las otras 3 features referidas al bullying:
- Bullied_on_school_property_in_last_12_months, 
- Bullied_not_on_school_property_in_last_12_months y 
- Cyber_Bullied_in_last_12_months; 
si alguno de estos es true, Bullied_in_last_12_months es true.


Obteniendo así la siguiente distribución de las dos clases: 
- (0) 'Not_bullied', si no han sufrido bullying en los últimos 12 meses y
- (1) 'Bullied', en caso contrario.

![Bullied_Distribution](./results/plots/Bullied_Distribution.png)

Distribución del bullying según sexo:

![Bullied_Sex_Distribution](./results/plots/Bullied_Sex_Distribution.png)

Distribución del bullying según si se sienten solos:

![Bullied_Felt_lonely_Distribution](./results/plots/Bullied_Felt_lonely_Distribution.png)

**Correlación con Cramer's V**

Índice de correlación de Cramer's V entre cada una de las variables y Bullied_in_past_12_months.

| Variable                                    | Cramer's V           |
| ------------------------------------------- | -------------------- |
| Felt_lonely                                 | 0.2726072072581656   |
| Physically_attacked                         | 0.23154321567889047  |
| Most_of_the_time_or_always_felt_lonely      | 0.21002465668566908  |
| Sex                                         | 0.1010648424129439   |
| Other_students_kind_and_helpful             | 0.13117254343399898  |
| Physical_fighting                           | 0.12324990208733567  |
| Parents_understand_problems                 | 0.11609267319217477  |
| Miss_school_no_permission                   | 0.08803480691919753  |
| Missed_classes_or_school_without_permission | 0.08438424530657992  |
| Close_friends                               | 0.07364435597446765  |
| Were_obese                                  | 0.022156982271592063 |
| Were_overweight                             | 0.021917650571953416 |
| Were_underweight                            | 0.02189826267397687  |
| Custom_Age                                  | 0.020020473152699978 |

Se probó agregar una feature Has_close_friends a partir de Close_friends pero empeoraba su valor de correlación de Cramer's V.

[Hacer matriz de correlación]

[Resultados de importance de RF]

Después de analizar los datos mediante Cramers'V y experimentar con diversas combinaciones de features con el objetivo de maximizar los resultados a través de la implementación de Random Forest, se ha determinado la utilización de las siguientes variables:

- Bullied_in_past_12_months
- Physically_attacked
- Physical_fighting
- Felt_lonely
- Sex

### Random Forest

Para la implementación del Random Forest, se emplearon los siguientes parámetros:

- `n_estimators=100`: El número de árboles en el bosque.

- `criterion='gini'`: La función para medir la calidad de un split.

- `max_depth=None`: La profundidad máxima del árbol. Si es `None`, los nodos se expanden hasta que todas las hojas sean puras o hasta que todas las hojas contengan menos de `min_samples_split` ejemplos.

- `min_samples_split=2`: El número mínimo de muestras necesario para dividir un nodo interno.

- `max_features='sqrt'`: El número de características a considerar al buscar la mejor división (sqrt(n_features)).

- `bootstrap=False`: Si se utilizan o no muestras de arranque al construir los árboles. Si es `False`, se utiliza todo el conjunto de datos para construir cada árbol.

- `class_weight={0: 1, 1: 1.5}`: Pesos asociados con las clases {Not bullied: 1, Bullied: 1.5}.

#### Train

**Matriz de confusión**

|              | Predicted Not bullied | Predicted Bullied |
| ------------ | ----------- | ----------- |
| **Actual Not bullied** | 16543       | 8119        |
| **Actual Bullied** | 6009        | 10217        |

**Reporte de la clasificación**

|                  | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| **Not bullied**      | 0.73      | 0.67   | 0.70     | 24662   |
| **Bullied**      | 0.56      | 0.63   | 0.59     | 16226   |
| **Accuracy**     |           |        | 0.65     | 40888   |
| **Macro Avg**    | 0.65      | 0.65   | 0.65     | 40888   |
| **Weighted Avg** | 0.66      | 0.65   | 0.66     | 40888   |

#### Test

**Matriz de confusión**

|              | Predicted Not bullied | Predicted Bullied |
| ------------ | ----------- | ----------- |
| **Actual Not bullied** | 4043        | 2055        |
| **Actual Bullied** | 1475        | 2649        |

**Reporte de la clasificación**

|                  | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| **Not bullied**      | 0.73      | 0.66   | 0.70     | 6098    |
| **Bullied**      | 0.56      | 0.64   | 0.60     | 4124    |
| **Accuracy**     |           |        | 0.65     | 10222   |
| **Macro Avg**    | 0.65      | 0.65   | 0.65     | 10222   |
| **Weighted Avg** | 0.66      | 0.65   | 0.66     | 10222   |

#### Gráficos

Generamos gráficos con el propósito de identificar la combinación óptima de la cantidad de árboles a utilizar en el modelo Random Forest, junto con los mejores pesos para las clases respectivas. Al analizar los resultados, observamos que no hubo una mejora significativa al aumentar el número de árboles, por lo que decidimos mantener n=100. En cuanto a los pesos de las clases, optamos por una opción equilibrada entre el accuracy y el recall, seleccionando Not bullied: 1 y Bullied: 1.5 (representados por la curva verde).

![rf_grid_search_accuracy.png](./results/plots/rf_grid_search_accuracy.png)

![rf_grid_search_recall.png](./results/plots/rf_grid_search_recall.png)

### Support Vector Machine

Para la implementación de SVM se emplearon los siguientes parámetros:

- `C=10`: El parámetro C controla la penalización por error en la clasificación. Un valor más alto de C hará que el modelo sea más estricto, tratando de clasificar correctamente todos los puntos de entrenamiento, pero puede llevar a overfitting.

- `kernel='rbf'`: Especifica el tipo de kernel a utilizar en el algoritmo. Se utiliza un kernel radial, que es comúnmente utilizado en problemas no lineales.

- `gamma= 0.001`: El parámetro gamma controla la amplitud de la función del kernel. Un valor bajo de gamma produce una función del kernel más suave, mientras que un valor alto puede llevar a overfitting.

- `probability=True`: Este parámetro habilita el cálculo de probabilidades de pertenencia a cada clase.

- `class_weight={0: 1, 1: 1.5}`: Pesos asociados con las clases {Not bullied: 1, Bullied: 1.5}.

#### Train

**Matriz de confusión**

|              | Predicted Not bullied | Predicted Bullied |
| ------------ | ----------- | ----------- |
| **Actual Not bullied** | 16664       | 7998        |
| **Actual Bullied** | 7543        | 8683        |

**Reporte de la clasificación**

|                  | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| **Not bullied**      | 0.69      | 0.68   | 0.68     | 24662   |
| **Bullied**      | 0.52      | 0.54   | 0.53     | 16226   |
| **Accuracy**     |           |        | 0.62     | 40888   |
| **Macro Avg**    | 0.60      | 0.61   | 0.60     | 40888   |
| **Weighted Avg** | 0.62      | 0.62   | 0.62     | 40888   |

#### Test

**Matriz de confusión**

|              | Predicted Not bullied | Predicted Bullied |
| ------------ | ----------- | ----------- |
| **Actual Not bullied** | 4161        | 1937        |
| **Actual Bullied** | 1882        | 2242        |

**Reporte de la clasificación**

|                  | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| **Not bullied**      | 0.69      | 0.68   | 0.69     | 6098    |
| **Bullied**      | 0.54      | 0.54   | 0.54     | 4124    |
| **Accuracy**     |           |        | 0.63     | 10222   |
| **Macro Avg**    | 0.61      | 0.61   | 0.61     | 10222   |
| **Weighted Avg** | 0.63      | 0.63   | 0.63     | 10222   |

#### Gráficos

Se generaron gráficos que representan el accuracy y el recall en función del factor de penalización de SVM, utilizando 15 divisiones diferentes. A partir de estos resultados, se seleccionó el valor de c=10, ya que demostró ser la elección que logra el mejor equilibrio entre ambas métricas.

![svm_boxplot_c_accuracy.png](./results/plots/svm_boxplot_c_accuracy.png)

![svm_boxplot_c_recall.png](./results/plots/svm_boxplot_c_recall.png)

## Análisis y discusión de resultados

### Comparacion de ambos algoritmos

![boxplot_rf_metrics](./results/plots/rf_boxplot_30.png)

![boxplot_svm_metrics](./results/plots/svm_boxplot_30.png)

### Learning Curves

![rf_learning_curve_error](./results/plots/rf_learning_curve_error.png)

## Conclusión

## Bibliografía

[https://www.ibm.com/mx-es/topics/random-forest#:~:text=El%20random%20forest%20es%20un,problemas%20de%20clasificaci%C3%B3n%20y%20regresi%C3%B3n]

