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

random_states = [43,18,76,92,5,61,29,80,12,50,8,37,64,3,97]
c_values = [0.1, 1, 10, 100]
kernel = ['linear', 'rbf']

results = {
    '0.1': {'Test_results': [], 'Train_results': []},
    '1': {'Test_results': [], 'Train_results': []},
    '10': {'Test_results': [], 'Train_results': []},
    '100': {'Test_results': [], 'Train_results': []},
}

accuracy_results = {
    '0.1': [],
    '1': [],
    '10': [],
    '100': [],
}

i = 1
""" for n in random_states:
    for c in c_values:
        start_time = time.time()
        print(i,"/", len(random_states) * len(c_values))
        i += 1
        print("Random state: ", n)
        print("C: ", c)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=n)
        # Create the model
        svm_model = SVC(kernel='rbf', C=c, random_state=n, probability=True, class_weight={0: 1, 1: 1.5}, gamma=0.001, verbose=True)
        
        # Train the model
        svm_model.fit(x_train, y_train)

        # Make predictions
        y_pred = svm_model.predict(x_train)

        # Evaluate the performance
        target_names = ['Not bullied', 'Bullied']
        train_result = Result( accuracy=accuracy_score(y_train, y_pred), precision=precision_score(y_train, y_pred), recall=recall_score(y_train, y_pred), f1=f1_score(y_train, y_pred), confusion_matrix=confusion_matrix(y_train, y_pred).tolist(), classification_report=classification_report(y_train, y_pred, target_names=target_names, zero_division=0))
        
        # Make predictions
        y_pred = svm_model.predict(x_test)

        # Evaluate the performance
        test_result = Result( accuracy=accuracy_score(y_test, y_pred), precision=precision_score(y_test, y_pred), recall=recall_score(y_test, y_pred), f1=f1_score(y_test, y_pred), confusion_matrix=confusion_matrix(y_test, y_pred).tolist(), classification_report=classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

        results[str(c)]['Train_results'].append(train_result.__dict__())
        results[str(c)]['Test_results'].append(test_result.__dict__())
        accuracy_results[str(c)].append(test_result.accuracy)

        print(confusion_matrix(y_test, y_pred), ": is the confusion matrix")
        print(accuracy_score(y_test, y_pred), ": is the accuracy score")
        print(precision_score(y_test, y_pred), ": is the precision score")
        print(recall_score(y_test, y_pred), ": is the recall score")
        print(f1_score(y_test, y_pred), ": is the f1 score")

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} s")
        print("----------------------------------------")
        print() """

with open("./results/json/svm_c_results_v2.json", "w") as json_file:
    json.dump(results, json_file, indent=2)

plt.boxplot([accuracy_results['0.1'], accuracy_results['1'], accuracy_results['10'], accuracy_results['100']], labels=['0.1', '1', '10', '100'])
plt.title("Accuracy según el factor de penalización (C) para SVM con 15 splits distintos")
plt.ylabel("Accuracy")
plt.xlabel("C")

plt.savefig("./results/plots/svm_boxplot_c_accuracy_v2.1.png")



# Load
with open("./results/json/svm_c_results_v2.json", "r") as json_file:
    data = json.load(json_file)

recall_results = [[result['recall'] for result in data[c]['Test_results']] for c in ['0.1', '1', '10', '100']]

plt.boxplot(recall_results, labels=['0.1', '1', '10', '100'])
plt.title("Recall según el factor de penalización (C) para SVM con 15 splits distintos")
plt.ylabel("Recall")
plt.xlabel("C")
plt.savefig("./results/plots/svm_boxplot_c_recall_v2.1.png")