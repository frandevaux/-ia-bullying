import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./data/Bullying_2018.csv",sep=';')
#print(df.head())
#df.info()

df.drop(['record', 'Were_underweight', 'Were_overweight', 'Were_obese'], axis=1, inplace=True)

# Replace empty strings with NaN
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
#print(df.isnull().sum())

# Convert "Yes" and "No" to 1 and 0
df.replace({'Yes': 1, 'No': 0}, inplace=True)

df['Sex'].fillna("Prefers not to answer", inplace=True)
df.dropna(inplace=True)

columns_to_check = ['Bullied_on_school_property_in_past_12_months', 'Bullied_not_on_school_property_in_past_12_months', 'Cyber_bullied_in_past_12_months']

# Find rows with null values in all specified columns
rows_with_all_null = df[df[columns_to_check].isnull().all(axis=1)]

# Drop the rows from the DataFrame
df.drop(rows_with_all_null.index, inplace=True)

# Join 3 columns into 1
df['Bullied_in_past_12_months'] = df[columns_to_check].apply(lambda row: 1 if row.any() == 1 else 0, axis=1)

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
plt.savefig('./results/plots/Bullied_in_past_12_months.png')"""

"""# Contar la cantidad de 0s y 1s para cada característica
counts = df[['Bullied_in_past_12_months', 'Bullied_on_school_property_in_past_12_months', 
              'Bullied_not_on_school_property_in_past_12_months', 'Cyber_bullied_in_past_12_months']].apply(pd.Series.value_counts)

# Crear un gráfico de barras apiladas
ax = counts.T.plot(kind='bar', stacked=True, color=['gray', 'black'])

# Configurar el gráfico
ax.set_ylabel('Count')
ax.set_title('Distribution of features about Bullying')
ax.legend(title='Bullied', labels=['0', '1'])

# Abreviar y mostrar los nombres horizontalmente
abbreviated_labels = ['Bullied', 'On_school', 'Not_on_school', 'Cyber_bullied']
ax.set_xticklabels(abbreviated_labels, rotation=0)

plt.savefig('./results/plots/Bullied_Distribution.png')"""


"""fig, ax = plt.subplots(figsize=(10, 6))

distribution = pd.crosstab(df['Sex'], df['Bullied_in_past_12_months'], margins=True, margins_name='Total')

# Crear un gráfico de barras apiladas con la cantidad total en lugar del porcentaje
ax = distribution[[0, 1]].plot(kind='bar', stacked=True, color=['gray', 'black'], ax=ax)

# Configurar el gráfico
ax.set_ylabel('Count')
ax.set_title('Distribution of Sex based on Bullied_in_past_12_months')
plt.legend(title='Bullied', labels=['0', '1'], loc='lower center')

abbreviated_labels = ['Female', 'Male', 'No answer', 'Total']
ax.set_xticklabels(abbreviated_labels, rotation=0)

plt.tight_layout()
plt.savefig('./results/plots/Bullied_Sex_Distribution.png')"""


"""fig, ax = plt.subplots(figsize=(10, 6))

distribution = pd.crosstab(df['Felt_lonely'], df['Bullied_in_past_12_months'], margins=True, margins_name='Total')

# Crear un gráfico de barras apiladas con la cantidad total en lugar del porcentaje
ax = distribution[[0, 1]].plot(kind='bar', stacked=True, color=['gray', 'black'], ax=ax)

# Configurar el gráfico
ax.set_ylabel('Count')
ax.set_title('Distribution of Felt_lonely based on Bullied_in_past_12_months')
plt.legend(title='Bullied', labels=['0', '1'], loc='lower center')

abbreviated_labels = ['Always', 'Most time', 'Never', 'Rarely', 'Sometimes', 'Total']
ax.set_xticklabels(abbreviated_labels, rotation=0)

plt.tight_layout()
plt.savefig('./results/plots/Bullied_Felt_lonely_Distribution.png')"""
