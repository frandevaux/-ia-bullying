import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import time
import pandas as pd

df = pd.read_csv("./results/fixed-Bullying_2018.csv",sep=';')
df= df[['Bullied_in_past_12_months',  'Physically_attacked', 'Physical_fighting', 'Felt_lonely', 'Sex']]

x = df.drop('Bullied_in_past_12_months', axis=1)
y = df['Bullied_in_past_12_months']

random_states = [43, 18, 76, 92, 5, 61, 29, 80, 12, 50]


# Función para ejecutar Random Forest y medir el tiempo
def run_random_forest(random_state):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)
    start_time = time.time()

    model = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight={0: 1, 1: 1.5})
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    end_time = time.time()
    return end_time - start_time

# Función para ejecutar SVM y medir el tiempo
def run_svm(random_state):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)
    start_time = time.time()

    model = SVC(kernel='rbf', C=10, gamma=0.001, class_weight={0: 1, 1: 1.5}, random_state=random_state, probability=True, verbose=True)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    end_time = time.time()
    return end_time - start_time

# Listas para almacenar los tiempos de ejecución
rf_runtimes = []
svm_runtimes = []
exec = 1

# Ejecutar ambos algoritmos y medir el tiempo en cada ejecución
for random_state in random_states:
    print("Ejecución:", exec, "de", len(random_states))
    print("Random Forest")
    rf_runtime = run_random_forest(random_state)
    print("SVM")
    svm_runtime = run_svm(random_state)
    rf_runtimes.append(rf_runtime)
    svm_runtimes.append(svm_runtime)
    exec += 1

# Crear un gráfico de líneas para comparar los tiempos
execution_numbers = np.arange(1, len(random_states) + 1)

plt.plot(execution_numbers, rf_runtimes, marker='o', label='Random Forest', color='orange')
plt.plot(execution_numbers, svm_runtimes, marker='o', label='SVM', color='blue')

plt.xlabel('Número de Ejecución')
plt.ylabel('Tiempo de Ejecución (segundos)')
plt.title('Comparación de Tiempos de Ejecución entre Random Forest y SVM')
plt.legend()
plt.savefig('./results/time_comparison.png')
