import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from result import Result
import random
import matplotlib.pyplot as plt

from result import Result

df = pd.read_csv("./results/fixed-Bullying_2018.csv",sep=';')

df= df[['Bullied_in_past_12_months',  'Physically_attacked', 'Physical_fighting', 'Felt_lonely', ]]

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Use LabelEncoder for ordinal categorical columns
label_encoder = LabelEncoder()
df[categorical_columns] = df[categorical_columns].apply(label_encoder.fit_transform)

# Use OneHotEncoder for nominal categorical columns
onehot_encoder = OneHotEncoder(sparse=False, drop='first')
onehot_encoded = onehot_encoder.fit_transform(df[categorical_columns])
df_onehot = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(categorical_columns))
df = pd.concat([df, df_onehot], axis=1)
df = df.drop(categorical_columns, axis=1)

# Split the dataset
x = df.drop('Bullied_in_past_12_months', axis=1)
y = df['Bullied_in_past_12_months']

small_random_states = [43, 18]
random_states = [43, 18 ,76,92,5,61,29,80,12,50,8,37,64,3,97]
# Calculate with different random_states

results_accuracy = []
results_precision = []
results_recall = []
results_f1 = []

target_names = ['Not bullied', 'Bullied']
test_results = [] 

for random_state in random_states:

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)

    # Create the model
    rf_model = RandomForestClassifier(n_estimators=250, random_state=random_state)

    # Train the model
    rf_model.fit(x_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(x_train)

    # Evaluate the performance
    train_result = Result( accuracy=accuracy_score(y_train, y_pred), precision=precision_score(y_train, y_pred), recall=recall_score(y_train, y_pred), f1=f1_score(y_train, y_pred), confusion_matrix=confusion_matrix(y_train, y_pred), classification_report=classification_report(y_train, y_pred, target_names=target_names, zero_division=0))
    
    
    
    
                          
    print("Evaluación con datos de entrenamiento")
    print(train_result.accuracy, ": is the accuracy score")
    print(train_result.precision, ": is the precision score")
    print(train_result.recall, ": is the recall score")
    print(train_result.f1, ": is the f1 score")
    print(train_result.confusion_matrix, ": is the confusion matrix")
    print(train_result.classification_report)
    print()
    
    
    # Make predictions
    y_pred = rf_model.predict(x_test)
    

    # Evaluate the performance
    test_result = Result( accuracy=accuracy_score(y_test, y_pred), precision=precision_score(y_test, y_pred), recall=recall_score(y_test, y_pred), f1=f1_score(y_test, y_pred), confusion_matrix=confusion_matrix(y_test, y_pred), classification_report=classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    
    test_results.append(test_result.__dict__())
    
    results_accuracy.append(train_result.accuracy)
    results_precision.append(train_result.precision)
    results_recall.append(train_result.recall)
    results_f1.append(train_result.f1)
    
    
    print("Evaluación con datos de prueba")
    print(test_result.accuracy, ": is the accuracy score")
    print(test_result.precision, ": is the precision score")
    print(test_result.recall, ": is the recall score")
    print(test_result.f1, ": is the f1 score")
    print(test_result.confusion_matrix, ": is the confusion matrix")
    print(test_result.classification_report)
    
""" with open("./results/rf_test_results.json", "w") as json_file:
    json.dump([result.__dict__ for result in test_results], json_file, indent=2) """
    
    

#Create boxplot with the results

plt.boxplot([results_accuracy, results_precision, results_recall, results_f1], labels=['Accuracy', 'Precision', 'Recall', 'F1'])
plt.title("Métricas de Random Forest con 15 combinaciones distintas de datos")
plt.ylabel("Score")
plt.xlabel("Metric")

plt.savefig("./results/boxplot_rf_metrics.png")

