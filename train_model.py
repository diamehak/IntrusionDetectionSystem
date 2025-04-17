# train_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset (adjust the path as needed)
data = pd.read_csv("../data/CICIDS2017_sample.csv")  # Adjust path if needed

# Drop nulls and irrelevant columns
data = data.dropna()
data = data.drop(columns=['Flow ID', 'Source IP', 'Destination IP', 'Timestamp'], errors='ignore')

# Encode the target variable (attack labels)
le = LabelEncoder()
data['Label'] = le.fit_transform(data['Label'])

# Feature and target variables
X = data.drop('Label', axis=1)
y = data['Label']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the trained model to a file
joblib.dump(model, "../model/ids_model.pkl")  # Save model
