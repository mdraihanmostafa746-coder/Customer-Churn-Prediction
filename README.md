# Customer Churn Prediction

A Machine Learning project that predicts whether a customer is likely to churn based on customer demographics, services, contract details, payment information, tenure, and billing data.

The project covers the complete Machine Learning workflow:

Data Collection → Data Cleaning → Exploratory Data Analysis → Feature Engineering → Preprocessing → Model Training → Model Evaluation → Cross Validation → Hyperparameter Tuning → Final Model Selection → Prediction

---

## 📌 Project Overview

Customer churn is a major problem for subscription-based businesses such as telecommunications companies.

If a company can identify customers who are likely to leave, it can take preventive actions such as:

- Personalized offers
- Customer support
- Contract upgrades
- Discounts
- Retention campaigns

The objective of this project is to build a classification model that predicts:

- `1` → Customer will churn
- `0` → Customer will not churn

---

## 🎯 Problem Statement

Build a Machine Learning classification system that predicts customer churn using historical customer data.

The model should not only achieve good accuracy but should also provide a reasonable balance between:

- Precision
- Recall
- F1 Score
- ROC-AUC

Since missing an actual churner can be costly for a business, Recall is an important metric in this project.

---

## 📊 Dataset

The project uses the Telco Customer Churn dataset.

The dataset contains information about:

### Customer Information

- Gender
- Senior Citizen
- Partner
- Dependents

### Services

- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies

### Account Information

- Tenure
- Contract
- Paperless Billing
- Payment Method

### Billing Information

- Monthly Charges
- Total Charges

### Target

- Churn

---

## 🧠 Machine Learning Problem

This is a:

**Binary Classification Problem**

Target:

```text
Churn

Yes → 1
No  → 0