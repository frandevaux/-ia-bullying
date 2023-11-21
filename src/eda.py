import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

df = pd.read_csv("./data/Bullying_2018.csv",sep=';')
#print(df.head())
#df.info()

# Replace empty strings with NaN
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
#print(df.isnull().sum())

# Convert "Yes" and "No" to 1 and 0
df.replace({'Yes': 1, 'No': 0}, inplace=True)

columns_to_check = ['Bullied_on_school_property_in_past_12_months', 'Bullied_not_on_school_property_in_past_12_months', 'Cyber_bullied_in_past_12_months']

# Find rows with null values in all specified columns
rows_with_all_null = df[df[columns_to_check].isnull().all(axis=1)]

# Drop the rows from the DataFrame
df.drop(rows_with_all_null.index, inplace=True)

# Join 3 columns into 1
df['Bullied_in_past_12_months'] = df[columns_to_check].apply(lambda row: 1 if row.any() == 1 else 0, axis=1)

# Drop columns with a high proportion of missing values
df.drop(['record', 'Bullied_on_school_property_in_past_12_months', 'Bullied_not_on_school_property_in_past_12_months', 'Cyber_bullied_in_past_12_months'], axis=1, inplace=True)

# Dropping na values from 'Custom_Age' column
df.dropna(subset=['Custom_Age'], inplace=True)

# Fill null values
df.fillna("Prefers not to answer", inplace=True)

# Write in txt all the features and their count
result = ""
for col in df.columns:
    col_count = str(df[col].value_counts())
    lines = col_count.split('\n')
    lines.pop()
    fixed_col_count = '\n'.join(lines)
    result += fixed_col_count + "\n" + "\n"

with open("./results/fixed-feature-count.txt", "w") as f:
    f.write(result)

"""for col in df.columns:
    col_count = str(df[col].value_counts())
    lines = col_count.split('\n')
    lines.pop()
    fixed_col_count = '\n'.join(lines)
    result += fixed_col_count + "\n" + "Null " + str(df[col].isnull().sum()) + "\n" + "\n"
print(result)"""

# Save in csv file
df.to_csv("./results/fixed-Bullying_2018.csv", index=False, sep=';')

# Plot the distribution of 'Bullied_in_past_12_months'
plt.figure(figsize=(8, 6))
sns.countplot(x='Bullied_in_past_12_months', data=df, palette='viridis')
plt.title('Distribution of Bullied_in_past_12_months')
plt.xlabel('Bullied_in_past_12_months')
plt.ylabel('Count')
plt.savefig('./results/Bullied_in_past_12_months.png')
