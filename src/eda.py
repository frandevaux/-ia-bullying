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

age_mapping = {
    '11 years old or younger': 11,
    '12 years old': 12,
    '13 years old': 13,
    '14 years old': 14,
    '15 years old': 15,
    '16 years old': 16,
    '17 years old': 17,
    '18 years old or older': 18,
    'Prefers not to answer': 15
}
df['Custom_Age'] = df['Custom_Age'].map(age_mapping)

physically_attacked_mapping = {
    '0 times': 0.0,
    '1 time': 1.0,
    '2 or 3 times': 2.5,
    '4 or 5 times': 4.5,
    '6 or 7 times': 6.5,
    '8 or 9 times': 8.5,
    '10 or 11 times': 10.5,
    '12 or more times': 12.0,
    'Prefers not to answer': 5.0
}
df['Physically_attacked'] = df['Physically_attacked'].map(physically_attacked_mapping)

physical_fighting_mapping = {
    '0 times': 0.0,
    '1 time': 1.0,
    '2 or 3 times': 2.5,
    '4 or 5 times': 4.5,
    '6 or 7 times': 6.5,
    '8 or 9 times': 8.5,
    '10 or 11 times': 10.5,
    '12 or more times': 12.0,
    'Prefers not to answer': 5.0
}
df['Physical_fighting'] = df['Physical_fighting'].map(physical_fighting_mapping)

close_friends_mapping = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3 or more': 3,
    'Prefers not to answer': 0
}
df['Close_friends'] = df['Close_friends'].map(close_friends_mapping)

miss_school_mapping = {
    '0 days': 0.0,
    '1 or 2 days': 1.5,
    '3 to 5 days': 4.0,
    '6 to 9 days': 7.5,
    '10 or more days': 10.0,
    'Prefers not to answer': 5.0
}
df['Miss_school_no_permission'] = df['Miss_school_no_permission'].map(miss_school_mapping)

felt_lonely_mapping = {
    'Never': 0,
    'Sometimes': 1,
    'Rarely': 2,
    'Most of the time': 3,
    'Always': 4
}
df['Felt_lonely'] = df['Felt_lonely'].map(felt_lonely_mapping)

other_students_mapping = {
    'Most of the time': 4,
    'Sometimes': 3,
    'Rarely': 2,
    'Always': 1,
    'Never': 0
}
df['Other_students_kind_and_helpful'] = df['Other_students_kind_and_helpful'].map(other_students_mapping)

parents_mapping = {
    'Always': 4,
    'Never': 0,
    'Rarely': 1,
    'Most of the time': 3,
    'Sometimes': 2
}
df['Parents_understand_problems'] = df['Parents_understand_problems'].map(parents_mapping)

sex_mapping = {
    'Female': 0,
    'Male': 1,
    'Prefers not to answer': 2
}
df['Sex'] = df['Sex'].map(sex_mapping)



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

""" # Plot the distribution of 'Bullied_in_past_12_months'
plt.figure(figsize=(8, 6))
sns.countplot(x='Bullied_in_past_12_months', data=df, palette=['#87CEEB', '#FFA500'])
plt.title('Distribution of Bullied_in_past_12_months')
plt.xlabel('Bullied_in_past_12_months')
plt.ylabel('Count')
plt.savefig('./results/plots/Bullied_in_past_12_months.png') """

"""# Contar la cantidad de 0s y 1s para cada característica
counts = df[['Bullied_in_past_12_months', 'Bullied_on_school_property_in_past_12_months', 
              'Bullied_not_on_school_property_in_past_12_months', 'Cyber_bullied_in_past_12_months']].apply(pd.Series.value_counts)

# Crear un gráfico de barras apiladas
ax = counts.T.plot(kind='bar', stacked=True, color=['#87CEEB', '#FFA500'])

# Configurar el gráfico
ax.set_ylabel('Cantidad')
ax.set_xlabel('Tipo de Bullying')
ax.set_title('Distribución de los tipos de Bullying')
ax.legend(labels=['No', 'Sí'], loc='lower center')

# Abreviar y mostrar los nombres horizontalmente
abbreviated_labels = ['De cualquier tipo', 'En la escuela', 'Fuera de la escuela', 'Ciberbullying']
ax.set_xticklabels(abbreviated_labels, rotation=0)

plt.savefig('./results/plots/Bullied_Distribution.png')"""


fig, ax = plt.subplots(figsize=(10, 6))

distribution = pd.crosstab(df['Sex'], df['Bullied_in_past_12_months'], margins=True, margins_name='Total')

# Crear un gráfico de barras apiladas con la cantidad total en lugar del porcentaje
ax = distribution[[0, 1]].plot(kind='bar', stacked=True, color=['#87CEEB', '#FFA500'], ax=ax)

# Configurar el gráfico
ax.set_ylabel('Cantidad')
ax.set_xlabel('Sexo')
ax.set_title('Cantidad de estudiantes que sufrieron bullying, de acuerdo a su sexo')
plt.legend(title='Sufre bullying', labels=['No', 'Sí'], loc='lower center')

abbreviated_labels = ['Femenino', 'Masculino', 'No responde', 'Total']
ax.set_xticklabels(abbreviated_labels, rotation=0)

plt.tight_layout()
plt.savefig('./results/plots/Bullied_Sex_Distribution.png')


fig, ax = plt.subplots(figsize=(10, 6))

distribution = pd.crosstab(df['Felt_lonely'], df['Bullied_in_past_12_months'], margins=True, margins_name='Total')

# Crear un gráfico de barras apiladas con la cantidad total en lugar del porcentaje
ax = distribution[[0, 1]].plot(kind='bar', stacked=True, color=['#87CEEB', '#FFA500'], ax=ax)

# Configurar el gráfico
ax.set_ylabel('Cantidad')
ax.set_xlabel('Se siente solo/a')
ax.set_title('Cantidad de estudiantes que sufrieron bullying, de acuerdo a si se sentían solos')
plt.legend(title='Sufre bullying', labels=['No', 'Sí'], loc='lower center')

abbreviated_labels = ['Siempre', 'Casi siempre', 'Nunca', 'Rara vez', 'A veces', 'Total']
ax.set_xticklabels(abbreviated_labels, rotation=0)

plt.tight_layout()
plt.savefig('./results/plots/Bullied_Felt_lonely_Distribution.png')
