import seaborn as sns; sns.set()
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("./results/fixed-Bullying_2018.csv",sep=';')

df= df[['Bullied_on_school_property_in_past_12_months','Sex', 'Felt_lonely', 'Close_friends', 'Other_students_kind_and_helpful', 'Parents_understand_problems']]

close_friends_mapping = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3 or more': 3,
    'Prefers not to answer': 0
}
df['Close_friends'] = df['Close_friends'].map(close_friends_mapping)

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
x = df.drop('Bullied_on_school_property_in_past_12_months', axis=1)
y = df['Bullied_on_school_property_in_past_12_months']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

param_grid_svm = {
    'C': [0.1, 1, 10, 100],                   
    'gamma': [1, 0.1, 0.01, 0.001],                
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'class_weight': ['balanced']                    
}

svmModel_grid = GridSearchCV(estimator=SVC(random_state=1234, probability=True), param_grid=param_grid_svm, verbose=1, cv=10, n_jobs=-1)
svmModel_grid.fit(x_train, y_train)

print(svmModel_grid.best_estimator_)

y_pred = svmModel_grid.predict(x_test)

print(confusion_matrix(y_test, y_pred), ": is the confusion matrix")
print(accuracy_score(y_test, y_pred), ": is the accuracy score")
print(precision_score(y_test, y_pred), ": is the precision score")
print(recall_score(y_test, y_pred), ": is the recall score")
print(f1_score(y_test, y_pred), ": is the f1 score")
