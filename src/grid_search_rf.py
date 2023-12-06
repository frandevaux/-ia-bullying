import seaborn as sns; sns.set()
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("./results/fixed-Bullying_2018.csv",sep=';')

df= df[['Bullied_in_past_12_months',  'Physically_attacked', 'Physical_fighting', 'Felt_lonely', 'Sex']]

# Split the dataset
x = df.drop('Bullied_in_past_12_months', axis=1)
y = df['Bullied_in_past_12_months']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

param_grid_svm = {
    'n_estimators': [100, 200, 250, 300, 400, 500],
    'criterion': ["gini", "entropy", "log_loss"],
    'max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'max_features': ["auto", "sqrt", "log2"],
    'bootstrap': [True, False],
    'class_weight': [{0: 1, 1: 1}, {0: 1, 1: 1.25}, {0: 1, 1: 1.5}, {0: 1, 1: 1.75}, {0: 1, 1: 2}]
}

model_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=0), param_grid=param_grid_svm, verbose=1, cv=10, n_jobs=-1)
model_grid.fit(x_train, y_train)

print(model_grid.best_estimator_)

y_pred = model_grid.predict(x_test)

print(confusion_matrix(y_test, y_pred), ": is the confusion matrix")
print(accuracy_score(y_test, y_pred), ": is the accuracy score")
print(precision_score(y_test, y_pred), ": is the precision score")
print(recall_score(y_test, y_pred), ": is the recall score")
print(f1_score(y_test, y_pred), ": is the f1 score")
