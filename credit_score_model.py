# Import Libraries
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------
print("Current folder:", os.getcwd())
print("File exists:", os.path.exists("dataset.csv"))

df = pd.read_csv("dataset.csv")

# Remove index column if present
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# -------------------------------------------------------
# Data Preprocessing
# -------------------------------------------------------

df["Saving accounts"] = df["Saving accounts"].fillna("unknown")
df["Checking account"] = df["Checking account"].fillna("unknown")

encoder = LabelEncoder()

for column in ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]:
    df[column] = encoder.fit_transform(df[column])

# -------------------------------------------------------
# Feature Engineering
# -------------------------------------------------------

# Create a binary target variable for demonstration
df["Credit Risk"] = (df["Credit amount"] > 5000).astype(int)

X = df.drop(["Credit Risk", "Credit amount"], axis=1)
y = df["Credit Risk"]

# -------------------------------------------------------
# Train-Test Split
# -------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -------------------------------------------------------
# Machine Learning Models
# -------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = []

# -------------------------------------------------------
# Model Training & Evaluation
# -------------------------------------------------------

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    results.append([name, accuracy, precision, recall, f1, roc])

    print("\n=======================================")
    print(name)
    print("=======================================")

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("ROC AUC  :", roc)

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

    # ROC Curve
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"{name} - ROC Curve")
    plt.show()

# -------------------------------------------------------
# Model Comparison
# -------------------------------------------------------

comparison = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC"
    ]
)

comparison = comparison.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n==============================")
print("Model Comparison")
print("==============================")

print(comparison)

best_model_name = comparison.iloc[0]["Model"]

print("\nBest Performing Model:", best_model_name)

# -------------------------------------------------------
# Feature Importance (Random Forest)
# -------------------------------------------------------

rf = models["Random Forest"]

importance = rf.feature_importances_

plt.figure(figsize=(10,5))
plt.bar(X.columns, importance)
plt.xticks(rotation=45)
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# Save Model
# -------------------------------------------------------

joblib.dump(rf, "credit_scoring_model.pkl")

print("\nModel saved successfully!")

comparison.to_csv("model_comparison.csv", index=False)

print("Comparison table saved!")

print("\nProject completed successfully!")
