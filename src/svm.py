import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

df = pd.read_csv("./results/fixed-Bullying_2018.csv",sep=';')

df= df[['Bullied_in_past_12_months', 'Sex', 'Felt_lonely', 'Close_friends', 'Other_students_kind_and_helpful', 'Parents_understand_problems', 'Physically_attacked', 'Physical_fighting', 'Miss_school_no_permission']]

"""age_mapping = {
    '11 years old or younger': 11,
    '12 years old': 12,
    '13 years old': 13,
    '14 years old': 14,
    '15 years old': 15,
    '16 years old': 16,
    '17 years old': 17,
    '18 years old or older': 18,
    'Prefers not to answer': 15
}
df['Custom_Age'] = df['Custom_Age'].map(age_mapping)"""

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Definimos la configuración del clasificador
clf = svm.SVC(kernel='rbf', C=1, gamma='auto', random_state=0, probability=True, verbose=True)

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
