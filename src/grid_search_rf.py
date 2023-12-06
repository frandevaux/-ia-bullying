import seaborn as sns; sns.set()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("./results/fixed-Bullying_2018.csv",sep=';')

df= df[['Bullied_in_past_12_months',  'Physically_attacked', 'Physical_fighting', 'Felt_lonely', 'Sex']]

# Split the dataset
x = df.drop('Bullied_in_past_12_months', axis=1)
y = df['Bullied_in_past_12_months']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

n_estimators = [100, 200, 250, 300, 400, 500]
class_weight = [{0: 1, 1: 1}, {0: 1, 1: 1.25}, {0: 1, 1: 1.5}, {0: 1, 1: 1.75}, {0: 1, 1: 2}]

param_grid_svm = {
    'n_estimators': n_estimators,
    'class_weight': class_weight
}

model_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=0), param_grid=param_grid_svm, verbose=1, cv=10, n_jobs=-1, scoring = ['accuracy', 'recall', 'precision', 'f1'], refit='accuracy')
model_grid.fit(x_train, y_train)

print(model_grid.best_estimator_)

y_pred = model_grid.predict(x_test)

print(confusion_matrix(y_test, y_pred), ": is the confusion matrix")
print(accuracy_score(y_test, y_pred), ": is the accuracy score")
print(precision_score(y_test, y_pred), ": is the precision score")
print(recall_score(y_test, y_pred), ": is the recall score")
print(f1_score(y_test, y_pred), ": is the f1 score")


def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_accuracy']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_accuracy']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Accuracy Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Accuracy', fontsize=16)
    ax.legend(fontsize=8, loc='center right', bbox_to_anchor=(1, 0.65))
    ax.grid('on')
    plt.savefig('./results/plots/rf_grid_search_accuracy.png')


    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_recall']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_recall']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Recall Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Recall', fontsize=16)
    ax.legend(loc='center right', fontsize=8)
    ax.grid('on')
    plt.savefig('./results/plots/rf_grid_search_recall.png')

# Calling Method 
plot_grid_search(model_grid.cv_results_, n_estimators, class_weight, 'N Estimators', 'Class Weight')