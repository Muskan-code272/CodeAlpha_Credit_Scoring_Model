# Credit Scoring Model

## Project Overview

The Credit Scoring Model is a Machine Learning project developed to predict an individual's creditworthiness using historical financial data. The project compares multiple classification algorithms and evaluates their performance using various metrics.

This project was completed as part of the **CodeAlpha Machine Learning Internship**.

## Objective

The objective of this project is to predict customer credit risk using machine learning classification algorithms and compare their performance using evaluation metrics.

## Dataset

The project uses the **German Credit Dataset**.

### Features Used

- Age
- Sex
- Job
- Housing
- Saving Accounts
- Checking Account
- Duration
- Purpose

### Target Variable

Since the dataset does not contain a predefined target variable, a binary **Credit Risk** column is created for demonstration purposes using the following condition:

```python
df["Credit Risk"] = (df["Credit amount"] > 5000).astype(int)
```

To avoid data leakage, the **Credit amount** column is removed from the feature set before training the models.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Joblib
- Git
- GitHub
- VS Code

## Machine Learning Models

The following classification algorithms are implemented:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

## Project Workflow

1. Load the dataset
2. Remove unnecessary columns
3. Handle missing values
4. Encode categorical variables using Label Encoding
5. Create the target variable
6. Split the dataset into training and testing sets
7. Train multiple machine learning models
8. Evaluate each model
9. Compare model performance
10. Save the trained model

## Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Classification Report
- Confusion Matrix
- ROC Curve

## Feature Engineering

The project includes:

- Missing value handling
- Label Encoding of categorical features
- Binary target variable creation
- Feature selection to prevent data leakage

## Project Structure

CodeAlpha_Credit_Scoring_Model/
│
├── credit_score_model.py
├── dataset.csv
├── credit_scoring_model.pkl
├── model_comparison.csv
├── README.md
└── requirements.txt
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Muskan-code272/CodeAlpha_Credit_Scoring_Model.git
```

### 2. Move into the project directory

```bash
cd CodeAlpha_Credit_Scoring_Model
```

### 3. Install dependencies

```bash
pip install pandas matplotlib scikit-learn joblib
```

or

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
python credit_score_model.py
```
## Output

The project generates:

- Dataset Information
- Missing Value Report
- Classification Report for each model
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix for each model
- ROC Curve for each model
- Feature Importance Graph
- Model Comparison Table
- Trained Model (`credit_scoring_model.pkl`)
- Model Comparison CSV (`model_comparison.csv`)

## Sample Output

The project displays:

- Confusion Matrix for Logistic Regression
- Confusion Matrix for Decision Tree
- Confusion Matrix for Random Forest
- ROC Curves for all models
- Random Forest Feature Importance
- Model Comparison Table sorted by Accuracy

## Future Improvements

- Hyperparameter tuning
- Cross-validation
- Model deployment using Flask or Streamlit
- Use a real credit risk target variable
- Web-based credit scoring application

## Author

**Muskan**

GitHub: https://github.com/Muskan-code272

Project Repository:
https://github.com/Muskan-code272/CodeAlpha_Credit_Scoring_Model

## License

This project is created for educational purposes as part of the **CodeAlpha Machine Learning Internship**.
