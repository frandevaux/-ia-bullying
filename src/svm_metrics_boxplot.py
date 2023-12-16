import seaborn as sns; sns.set()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from result import Result

df = pd.read_csv("./results/fixed-Bullying_2018.csv",sep=';')

df= df[['Bullied_in_past_12_months',  'Physically_attacked', 'Physical_fighting', 'Felt_lonely', 'Sex', 'Miss_school_no_permission', 'Other_students_kind_and_helpful', 'Parents_understand_problems']]

# Split the dataset
x = df.drop('Bullied_in_past_12_months', axis=1)
y = df['Bullied_in_past_12_months']

random_states = [43, 18, 76, 92, 5, 61, 29, 80, 12, 50, 8, 37, 64, 3, 97, 22, 40, 55, 89, 14, 72, 33, 95, 7, 49, 81, 26, 68, 45, 11, 93]

results = []
i = 1
for n in random_states:
    start_time = time.time()
    print(i,"/", len(random_states))
    i += 1
    print("Random state: ", n)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=n)
    # Create the model
    model = SVC(kernel='rbf', C=10, random_state=n, probability=True, class_weight={0: 1, 1: 1.5}, gamma=0.001, verbose=True)
    
    # Train the model
    model.fit(x_train, y_train)

    # Make predictions
    y_pred = model.predict(x_train)

    # Evaluate the performance
    target_names = ['Not bullied', 'Bullied']
    train_result = Result(accuracy=accuracy_score(y_train, y_pred), precision=precision_score(y_train, y_pred), recall=recall_score(y_train, y_pred), f1=f1_score(y_train, y_pred), confusion_matrix=confusion_matrix(y_train, y_pred).tolist(), classification_report=classification_report(y_train, y_pred, target_names=target_names, zero_division=0))
    
    # Make predictions
    y_pred = model.predict(x_test)

    # Evaluate the performance
    test_result = Result(accuracy=accuracy_score(y_test, y_pred), precision=precision_score(y_test, y_pred), recall=recall_score(y_test, y_pred), f1=f1_score(y_test, y_pred), confusion_matrix=confusion_matrix(y_test, y_pred).tolist(), classification_report=classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    result = {'Train': train_result.__dict__(), 'Test': test_result.__dict__()}
    results.append(result)

    print(confusion_matrix(y_test, y_pred), ": is the confusion matrix")
    print(accuracy_score(y_test, y_pred), ": is the accuracy score")
    print(precision_score(y_test, y_pred), ": is the precision score")
    print(recall_score(y_test, y_pred), ": is the recall score")
    print(f1_score(y_test, y_pred), ": is the f1 score")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} s")
    print("----------------------------------------")
    print()

with open("./results/json/svm_30_results_v2.json", "w") as json_file:
    json.dump(results, json_file, indent=2)

accuracy_results = [result['Test']['accuracy'] for result in results]
precision_results = [result['Test']['precision'] for result in results]
recall_results = [result['Test']['recall'] for result in results]
f1_results = [result['Test']['f1'] for result in results]

plt.boxplot([accuracy_results, precision_results, recall_results, f1_results], labels=['Accuracy', 'Precision', 'Recall', 'F1'])
plt.xlabel('Métrica')
plt.ylabel('Valor')
plt.title('Métricas en los resultados de test de SVM con 30 splits distintos')

plt.savefig("./results/plots/svm_boxplot_30_v2.png") 
