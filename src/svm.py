import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
import time
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

start_time = time.time()

df = pd.read_csv("./results/fixed-Bullying_2018.csv",sep=';')

df= df[['Bullied_in_past_12_months',  'Physically_attacked', 'Physical_fighting', 'Felt_lonely', 'Sex']]

# Split the dataset
x = df.drop('Bullied_in_past_12_months', axis=1)
y = df['Bullied_in_past_12_months']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#sampler = RandomOverSampler(sampling_strategy='auto', random_state=0)
#x_train, y_train = sampler.fit_resample(x_train, y_train)

# Definimos la configuración del clasificador
clf = svm.SVC(kernel='rbf', C=0.1, gamma=0.001, class_weight={0: 1, 1: 1.5}, random_state=0, probability=True, verbose=True)

# Entrenamos el clasificador con los datos de entrenamiento
clf.fit(x_train, y_train)

y_train_pred = clf.predict(x_train)
y_pred = clf.predict(x_test)

# Train
print("Train")
print(confusion_matrix(y_train, y_train_pred), ": is the confusion matrix")
print(accuracy_score(y_train, y_train_pred), ": is the accuracy score")
print(precision_score(y_train, y_train_pred), ": is the precision score")
print(recall_score(y_train, y_train_pred), ": is the recall score")
print(f1_score(y_train, y_train_pred), ": is the f1 score")
target_names = ['Not bullied', 'Bullied']
print(classification_report(y_train, y_train_pred, target_names=target_names, zero_division=0))
print("")

# Test
print("Test")
print(confusion_matrix(y_test, y_pred), ": is the confusion matrix")
print(accuracy_score(y_test, y_pred), ": is the accuracy score")
print(precision_score(y_test, y_pred), ": is the precision score")
print(recall_score(y_test, y_pred), ": is the recall score")
print(f1_score(y_test, y_pred), ": is the f1 score")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

"""# Identifica los índices de los vectores de soporte
support_indices = clf.support_

# Extrae las variables asociadas a los vectores de soporte
support_features = x_train.iloc[support_indices, :]

# Calcula la importancia relativa basada en la frecuencia de las variables entre los vectores de soporte
feature_importance = support_features.mean(axis=0)

# Crea un DataFrame para visualizar las importancias
importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)"""


end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} s")
