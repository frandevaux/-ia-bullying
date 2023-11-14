import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("./data/Bullying_2018.csv",sep=';')
#print(df.head())
#df.info()

# Replace empty strings with NaN
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
print(df.isnull().sum())

'''
for col in df.columns:
    percent_missing_value = (df[col].isnull().sum()/df.shape[0]) * 100
    print("Percent of missing values for the column ", col, " is ", percent_missing_value)
'''

result = ""

for col in df.columns:
    result += str(df[col].value_counts()) + "\n" + "\n"

# Save in txt file
with open("./results/raw-feature-count.txt", "w") as f:
    f.write(result)
