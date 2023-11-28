import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import random
import matplotlib.pyplot as plt

df = pd.read_csv("./results/fixed-Bullying_2018.csv",sep=';')

df= df[['Bullied_on_school_property_in_past_12_months',  'Physically_attacked', 'Felt_lonely', 'Close_friends', 'Miss_school_no_permission', 'Missed_classes_or_school_without_permission', ]]

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Use LabelEncoder for ordinal categorical columns
label_encoder = LabelEncoder()
df[categorical_columns] = df[categorical_columns].apply(label_encoder.fit_transform)

# Use OneHotEncoder for nominal categorical columns
onehot_encoder = OneHotEncoder(sparse=False, drop='first')
onehot_encoded = onehot_encoder.fit_transform(df[categorical_columns])
df_onehot = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(categorical_columns))
df = pd.concat([df, df_onehot], axis=1)
df = df.drop(categorical_columns, axis=1)

# Split the dataset
x = df.drop('Bullied_on_school_property_in_past_12_months', axis=1)
y = df['Bullied_on_school_property_in_past_12_months']


# Calculate with different random_states

results = []

for _ in range(15):


    random_state = random.randint(0, 100)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)

    # Create the model
    rf_model = RandomForestClassifier(n_estimators=250, random_state=random_state)

    # Train the model
    rf_model.fit(x_train, y_train)

    """# Make predictions
    y_pred = rf_model.predict(x_train)

    # Evaluate the performance
    print("Evaluación con datos de entrenamiento")
    print("Accuracy:", accuracy_score(y_train, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred))
    print("Classification Report:\n", classification_report(y_train, y_pred))
    print()"""
    # Make predictions
    y_pred = rf_model.predict(x_test)

    # Evaluate the performance
    print("Evaluación con datos de prueba")
    accuracy = accuracy_score(y_test, y_pred)
    results.append(accuracy)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

#Create boxplot with the results

plt.boxplot(results)
plt.title("Accuracy de Random Forest con 15 combinaciones de datos de prueba distintas")
plt.show()
plt.savefig("./results/accuracy.png")