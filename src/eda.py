import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./data/Bullying_2018.csv",sep=';')
#print(df.head())
#df.info()

# Replace empty strings with NaN
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
#print(df.isnull().sum())

# Convert "Yes" and "No" to 1 and 0
df.replace({'Yes': 1, 'No': 0}, inplace=True)

# Drop columns with a high proportion of missing values
df.drop(['record'], axis=1, inplace=True)

# Dropping na values from 'Custom_Age' column
df.dropna(subset=['Custom_Age'], inplace=True)

df.dropna(inplace=True)

"""# Fill null values
df.fillna("Prefers not to answer", inplace=True)"""

"""# Create a new feature 'Has_close_friends' based on 'Close_Friends'
df['Has_close_friends'] = df['Close_friends'].apply(lambda x: 1 if x != '0' else 0)

# Create a new feature 'Physically_attacked_num' based on 'Physically_attacked'
mapping = {
    '0 times': 0,
    '1 time': 1,
    '2 or 3 times': 2.5,
    '4 or 5 times': 4.5,
    '6 or 7 times': 6.5,
    'Prefers not to answer': 0,
    '8 or 9 times': 8.5,
    '10 or 11 times': 10.5
}

df['Physically_attacked_num'] = df['Physically_attacked'].map(mapping)"""

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

"""# Plot the distribution of 'Bullied_in_past_12_months'
plt.figure(figsize=(8, 6))
sns.countplot(x='Bullied_in_past_12_months', data=df, palette='viridis')
plt.title('Distribution of Bullied_in_past_12_months')
plt.xlabel('Bullied_in_past_12_months')
plt.ylabel('Count')
plt.savefig('./results/Bullied_in_past_12_months.png')"""
