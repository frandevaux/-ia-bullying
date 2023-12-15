import matplotlib.pyplot as plt

# Datos
variables = ["Felt_lonely", "Physically_attacked", "Most_of_the_time_or_always_felt_lonely", 
             "Sex", "Other_students_kind_and_helpful", "Physical_fighting", 
             "Parents_understand_problems", "Miss_school_no_permission", 
             "Missed_classes_or_school_without_permission", "Close_friends", 
             "Were_obese", "Were_overweight", "Were_underweight", "Custom_Age"]

cramers_v = [0.2726072072581656, 0.23154321567889047, 0.21002465668566908, 
             0.1010648424129439, 0.13117254343399898, 0.12324990208733567, 
             0.11609267319217477, 0.08803480691919753, 0.08438424530657992, 
             0.07364435597446765, 0.022156982271592063, 0.021917650571953416, 
             0.02189826267397687, 0.020020473152699978]

# Crear un gráfico de barras
plt.figure(figsize=(10, 6))
plt.barh(variables, cramers_v, color='skyblue')
plt.xlabel("Cramer's V", fontsize=16)
plt.title("Asociación entre variables (Cramer's V)", fontsize=16)
plt.grid(axis='x')

# Mostrar los valores en las barras con un tamaño de fuente más grande
for index, value in enumerate(cramers_v):
    plt.text(value, index, f"{value:.3f}", va='center', fontsize=16)

plt.show()
