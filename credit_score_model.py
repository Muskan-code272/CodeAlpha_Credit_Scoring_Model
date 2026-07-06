# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
import os

print("Current folder:", os.getcwd())
print("File exists:", os.path.exists("dataset.csv"))

df = pd.read_csv("dataset.csv")
#Remove the index column if it exists

df.drop("Unnamed: 0", axis=1, inplace=True)
print(df.head())

# Display first 5 rows
print(df.head())

# Display dataset information
print(df.info())

# Check missing values
print(df.isnull().sum())

# Fill missing values
df["Saving accounts"] = df["Saving accounts"].fillna("unknown")
df["Checking account"] = df["Checking account"].fillna("unknown")

# Convert categorical columns to numbers
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

for column in ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]:
    df[column] = encoder.fit_transform(df[column])

# Create target variable
df["Credit Risk"] = (df["Credit amount"] > 5000).astype(int)

# Features and target
X = df.drop("Credit Risk", axis=1)
y = df["Credit Risk"]

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title("Confusion Matrix")
plt.show()

importance = model.feature_importances_

features = X.columns

plt.figure(figsize=(10,5))
plt.bar(features, importance)
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()