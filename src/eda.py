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

# Convert "Yes" and "No" to 1 and 0
df.replace({'Yes': "1", 'No': "0"}, inplace=True)

result = ""
"""
for col in df.columns:
    col_count = str(df[col].value_counts())
    lines = col_count.split('\n')
    lines.pop()
    fixed_col_count = '\n'.join(lines)
    result += col + "\n" + fixed_col_count + "\n" + "Null " + str(df[col].isnull().sum()) + "\n" + "\n"

# Save in txt file
with open("./results/fixed-feature-count.txt", "w") as f:
    f.write(result)"""

df['Bullied_in_past_12_months'] = df[['Bullied_on_school_property_in_past_12_months', 'Bullied_not_on_school_property_in_past_12_months', 'Cyber_bullied_in_past_12_months']].apply(lambda row: 1 if row.any() == 1 else 0, axis=1)

for col in df.columns:
    col_count = str(df[col].value_counts())
    lines = col_count.split('\n')
    lines.pop()
    fixed_col_count = '\n'.join(lines)
    result += col + "\n" + fixed_col_count + "\n" + "Null " + str(df[col].isnull().sum()) + "\n" + "\n"
print(result)

# Save in csv file
df.to_csv("./results/fixed-Bullying_2018.csv", index=False)