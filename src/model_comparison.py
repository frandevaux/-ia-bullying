import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import random
import matplotlib.pyplot as plt


# Random Forest

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Create the model
rf_model = RandomForestClassifier(n_estimators=250, random_state=0)

# Train the model
rf_model.fit(x_train, y_train)

# Make predictions
y_pred = rf_model.predict(x_train)

# Evaluate the performance
print("Evaluación con datos de entrenamiento")
print("Accuracy:", accuracy_score(y_train, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred))
print("Classification Report:\n", classification_report(y_train, y_pred))
print()
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
rf_train_results = [accuracy_score(y_train, y_pred), precision_score(y_train, y_pred), recall_score(y_train, y_pred), f1_score(y_train, y_pred)]

# Make predictions
y_pred = rf_model.predict(x_test)

# Evaluate the performance
print("Evaluación con datos de prueba")
recall = recall_score(y_test, y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
rf_test_results = [accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)]



# SVM

df = pd.read_csv("./results/fixed-Bullying_2018.csv",sep=';')
df= df[['Bullied_in_past_12_months', 'Sex', 'Felt_lonely', 'Close_friends', 'Other_students_kind_and_helpful', 'Parents_understand_problems', 'Physically_attacked', 'Physical_fighting', 'Miss_school_no_permission']]

physically_attacked_mapping = {
    '0 times': 0.0,
    '1 time': 1.0,
    '2 or 3 times': 2.5,
    '4 or 5 times': 4.5,
    '6 or 7 times': 6.5,
    '8 or 9 times': 8.5,
    '10 or 11 times': 10.5,
    '12 or more times': 12.0,
    'Prefers not to answer': 5.0
}
df['Physically_attacked'] = df['Physically_attacked'].map(physically_attacked_mapping)

physical_fighting_mapping = {
    '0 times': 0.0,
    '1 time': 1.0,
    '2 or 3 times': 2.5,
    '4 or 5 times': 4.5,
    '6 or 7 times': 6.5,
    '8 or 9 times': 8.5,
    '10 or 11 times': 10.5,
    '12 or more times': 12.0,
    'Prefers not to answer': 5.0
}
df['Physical_fighting'] = df['Physical_fighting'].map(physical_fighting_mapping)

close_friends_mapping = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3 or more': 3,
    'Prefers not to answer': 0
}
df['Close_friends'] = df['Close_friends'].map(close_friends_mapping)

miss_school_mapping = {
    '0 days': 0.0,
    '1 or 2 days': 1.5,
    '3 to 5 days': 4.0,
    '6 to 9 days': 7.5,
    '10 or more days': 10.0,
    'Prefers not to answer': 5.0
}
df['Miss_school_no_permission'] = df['Miss_school_no_permission'].map(miss_school_mapping)


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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Definimos la configuración del clasificador
clf = svm.SVC(kernel='rbf', C=0.1, gamma=0.001, class_weight='balanced', random_state=0, probability=True, verbose=True)

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
target_names = ['Class 0', 'Class 1']
print(classification_report(y_train, y_train_pred, target_names=target_names, zero_division=0))
print("")
svm_train_results = [accuracy_score(y_train, y_train_pred), precision_score(y_train, y_train_pred), recall_score(y_train, y_train_pred), f1_score(y_train, y_train_pred)]

# Test
print("Test")
print(confusion_matrix(y_test, y_pred), ": is the confusion matrix")
print(accuracy_score(y_test, y_pred), ": is the accuracy score")
print(precision_score(y_test, y_pred), ": is the precision score")
print(recall_score(y_test, y_pred), ": is the recall score")
print(f1_score(y_test, y_pred), ": is the f1 score")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
svm_test_results = [accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)]





# Posiciones de las barras
bar_positions_rf = np.arange(len(metrics))
bar_positions_svm = [pos + 0.2 for pos in bar_positions_rf]
bar_positions_test_rf = [pos + 0.2 for pos in bar_positions_svm]
bar_positions_test_svm = [pos + 0.2 for pos in bar_positions_test_rf]

# Tamaño de las barras
bar_width = 0.2

# Crear el gráfico de barras
plt.bar(bar_positions_rf, rf_train_results, width=bar_width, label='Random Forest Train', color='#0e9920')
plt.bar(bar_positions_svm, svm_train_results, width=bar_width, label='Support Vector Machine Train', color='#bfb80d')
plt.bar(bar_positions_test_rf, rf_test_results, width=bar_width, label='Random Forest Test', color='#49df5d')
plt.bar(bar_positions_test_svm, svm_test_results, width=bar_width, label='Support Vector Machine Test', color='#f1eb32')

# Configurar el gráfico
plt.xlabel('Metric')
plt.ylabel('Score')
plt.title('Comparison of Model Performance Metrics')
plt.xticks([pos + bar_width*1.5 for pos in bar_positions_rf], metrics)
plt.legend()
plt.ylim(0, 1)

plt.savefig('./results/plots/Model_Metrics_Comparison.png')
