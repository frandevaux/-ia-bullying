import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBRFClassifier
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("./data/Bullying_2018.csv",sep=';')
#print(df.head())
#df.info()

data = data[['Bullied_on_school_property_in_past_12_months', 'Cyber_bullied_in_past_12_months', 'Custom_Age','Sex',
            'Felt_lonely', 'Close_friends', 'Other_students_kind_and_helpful', 'Parents_understand_problems', 'Were_overweight']]

data['Custom_Age'] = data['Custom_Age'].str.replace('years old', '')
#data['Physically_attacked'] = data['Physically_attacked'].str.replace('times', '')
data['Close_friends'] = data['Close_friends'].str.replace(' or more', '')

data['Bullied_on_school_property_in_past_12_months'] = data['Bullied_on_school_property_in_past_12_months'].replace({'No': 0, 'Yes': 1})
data['Cyber_bullied_in_past_12_months'] = data['Cyber_bullied_in_past_12_months'].replace({'No': 0, 'Yes': 1})
data['Were_overweight'] = data['Were_overweight'].replace({'No': 0, 'Yes': 1})
data['Sex'] = data['Sex'].replace({'Male': 0, 'Female': 1})

data.replace('', np.nan, inplace = True)
data = data.replace(' ', np.nan)

data = data.dropna()

prefix_col = ['Felt_lone', 'Other_students_kind', 'Parents_understand_problems']
dummy_col = ['Felt_lonely', 'Other_students_kind_and_helpful', 'Parents_understand_problems']
data = pd.get_dummies(data, prefix = prefix_col, columns = dummy_col)

data = data.replace(' ', np.nan)

data = data.dropna()

data = data[~data['Custom_Age'].str.contains("18  or older")]
data = data[~data['Custom_Age'].str.contains("11  or younger")]

data['Bullied_on_school_property_in_past_12_months'] = pd.to_numeric(data['Bullied_on_school_property_in_past_12_months'])
data['Cyber_bullied_in_past_12_months'] = pd.to_numeric(data['Cyber_bullied_in_past_12_months'])
data['Custom_Age'] = pd.to_numeric(data['Custom_Age'])
data['Sex'] = pd.to_numeric(data['Sex'])
#data['Physically_attacked'] = pd.to_numeric(data['Physically_attacked'])
data['Close_friends'] = pd.to_numeric(data['Close_friends'])
data['Were_overweight'] = pd.to_numeric(data['Were_overweight'])

X = data.drop(columns = ['Bullied_on_school_property_in_past_12_months'])
y = data['Bullied_on_school_property_in_past_12_months']

sm = SMOTE()
X, y = sm.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
xg = XGBRFClassifier()
xg.fit(X_train, y_train)
y_pred = xg.predict(X_test)
train_accuracy = xg.score(X_train, y_train)
test_accuracy = xg.score(X_test, y_test)
print( "Train accuracy:", train_accuracy)
print( "Test accuracy:", test_accuracy)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
train_accuracy = rf.score(X_train, y_train)
test_accuracy = rf.score(X_test, y_test)
print('')
print( "Train accuracy:", train_accuracy)
print( "Test accuracy:", test_accuracy)