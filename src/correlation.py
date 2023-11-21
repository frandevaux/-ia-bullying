import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

df = pd.read_csv("./results/fixed-Bullying_2018.csv",sep=';')

categorical_columns = ['Custom_Age', 'Sex', 'Physically_attacked', 'Physical_fighting', 'Felt_lonely',
                        'Close_friends', 'Miss_school_no_permission', 'Other_students_kind_and_helpful',
                        'Parents_understand_problems', 'Most_of_the_time_or_always_felt_lonely',
                        'Missed_classes_or_school_without_permission', 'Were_underweight', 'Were_overweight',
                        'Were_obese', 'Bullied_in_past_12_months']

for col in categorical_columns:
    # Construye una tabla de contingencia
    contingency_table = pd.crosstab(df['Bullied_in_past_12_months'], df[col])

    # Aplica la prueba de chi-cuadrado
    chi2, p, _, _ = chi2_contingency(contingency_table)

    # Tamaño de la muestra
    n = len(df)

    # Número de categorías en cada variable
    k = len(contingency_table.columns)
    r = len(contingency_table.index)

    # Calcula Cramer's V
    cramers_v = np.sqrt(chi2 / (n * min(k - 1, r - 1)))

    print(f"Variable: {col}")
    print(f"Cramer's V: {cramers_v}")
    print("\n")
