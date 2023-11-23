import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

df = pd.read_csv("./results/fixed-Bullying_2018.csv",sep=';')

df.drop(['Were_underweight', 'Were_overweight', 'Were_obese',  'Missed_classes_or_school_without_permission', 'Close_friends', 'Most_of_the_time_or_always_felt_lonely'], axis=1, inplace=True)

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
clf = svm.SVC(kernel='rbf')

# Entrenamos el clasificador con los datos de entrenamiento
clf.fit(x_train, y_train)

# Predecimos los valores de los datos de prueba
score = clf.score(x_test, y_test)
print(score)

ypred = clf.predict(x_test)
matriz = confusion_matrix(y_test,ypred)

plot_confusion_matrix(conf_mat=matriz, figsize=(6,6), show_normed=False)
plt.tight_layout()
