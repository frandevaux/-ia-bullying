import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

df = pd.read_csv("./results/fixed-Bullying_2018.csv",sep=';')

df= df[['Bullied_in_past_12_months',  'Physically_attacked', 'Physical_fighting', 'Felt_lonely', 'Sex']]
print(len(df) * 0.8)

# Split the dataset
x = df.drop('Bullied_in_past_12_months', axis=1)
y = df['Bullied_in_past_12_months']

train_sizes, train_scores, validation_scores = learning_curve(estimator = RandomForestClassifier(n_estimators=100, random_state=0, class_weight={0: 1, 1: 1.5}), X = x, y = y, scoring = 'accuracy', train_sizes = [1, 10, 100, 500, 1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40888], cv = 10)

print('Training scores:\n\n', train_scores)
print('\n', '-' * 70)
print('\nValidation scores:\n\n', validation_scores)

train_scores_mean = train_scores.mean(axis = 1)
validation_scores_mean = validation_scores.mean(axis = 1)
print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
print('\n', '-' * 20)
print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))

plt.plot(train_sizes, train_scores_mean, label = 'Training')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation')
plt.ylabel('Accuracy', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for a random forest model', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(0,1)
plt.savefig('./results/plots/rf_learning_curve.png')
