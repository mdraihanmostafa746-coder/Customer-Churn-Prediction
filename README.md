# Customer Churn Prediction

A Machine Learning project that analyzes customer data and predicts whether a telecom customer is likely to leave the service (Churn) using classification algorithms.

## Project Overview

Customer churn is an important challenge for telecom companies. Understanding customer behavior and identifying customers who are likely to leave can help businesses improve customer retention.

In this project, I performed Exploratory Data Analysis (EDA), data preprocessing, model training, and evaluation to predict customer churn.

## Dataset

- **Dataset:** Telco Customer Churn
- **Total Records:** 7,043
- **Total Features:** 20
- **Target Variable:** Churn

The dataset contains customer information such as:
- Customer demographics
- Account information
- Services subscribed
- Contract details
- Payment methods
- Monthly and total charges

## Technologies & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

## Project Workflow

1. Data Loading and Understanding
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature and Target Preparation
5. Train-Test Split
6. Feature Scaling and Encoding
7. Model Training
8. Model Evaluation
9. Cross-Validation
10. Hyperparameter Tuning

## Machine Learning Models

The following classification algorithms were implemented and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

## Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

Based on the original notebook's test-set evaluation, Logistic Regression achieved the highest F1 Score among the five tested models.

## Web Application

A Streamlit web application is being developed to make customer churn prediction interactive.

Planned features include:

- Customer information input
- Churn prediction
- Churn probability
- Model performance metrics
- Interactive user interface

## Project Structure

```text
Customer-Churn-Prediction/
│
├── app.py
├── train_model.py
├── model.pkl
├── requirements.txt
├── README.md
│
└── data/
    └── customer_churn.csv
```

*Note: The project structure will be updated as development progresses.*

## Run Locally

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
```

Navigate to the project directory:

```bash
cd Customer-Churn-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Project Status

🚧 Complete Streamlit application deployment.

## Author

**Md Raihan Mostafa**

B.Sc. Data Science Student | Aspiring AI Engineer

