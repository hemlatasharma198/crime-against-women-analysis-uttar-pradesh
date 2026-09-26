# Crime Against Women — Uttar Pradesh

An interactive data analytics and machine learning application for
analyzing crime against women in Uttar Pradesh.

The project combines historical data analysis, district-level crime
profiling, and machine learning-based next-year crime forecasting
through a Streamlit dashboard.

---

## Project Overview

This project analyzes historical crime data related to women in
Uttar Pradesh and provides an interactive interface to explore
patterns across years, districts, and crime categories.

The application has two major analytical layers:

1. **District Risk Profile**
   - Compares the historical crime burden of districts.
   - Identifies the top crime categories for a selected district.
   - Shows recent crime trends.
   - Provides preventive focus areas based on the identified categories.

2. **Next-Year Crime Forecast**
   - Uses previous-year district-level crime data.
   - Predicts next-year total crime using a Random Forest Regressor.

---

## Features

### Crime Analysis

Explore historical crime patterns using:

- Year-wise crime analysis
- Crime category distribution
- District-wise crime comparison
- Top districts by reported crime

### District Risk Profile

Select a district and view:

- Historical Crime Indicator
- Historical Burden Band
- Overall crime trend
- Top 3 crime categories
- Relative crime indicators
- Preventive focus areas

> The Historical Crime Indicator is a relative historical measure
> based on reported crime counts. It is not a probability of future
> crime occurrence.

### Crime Prediction

The application uses previous-year crime information to forecast
next-year total crime for a district.

### Model Performance

The dashboard provides:

- MAE
- RMSE
- R² Score
- Actual vs Predicted comparison
- Feature Importance

### Key Insights

The application summarizes important findings from:

- Crime categories
- Year-wise trends
- District-wise crime
- Machine learning analysis

---

## Crime Categories

The dataset contains the following crime categories:

- Rape
- Kidnapping / Abduction
- Dowry Deaths
- Assault to Outrage Modesty
- Insult to Modesty of Women
- Cruelty by Husband or Relatives
- Importation of Girls

---

## Machine Learning

### Model Used

**Random Forest Regressor**

The model predicts next-year total crime using previous-year
district-level crime information.

### Features

The model uses:

- Previous-year Rape
- Previous-year Kidnapping / Abduction
- Previous-year Dowry Deaths
- Previous-year Assault to Outrage Modesty
- Previous-year Insult to Modesty
- Previous-year Cruelty by Husband / Relatives
- Previous-year Importation of Girls
- Previous-year Total Crime

A time-based train-test split was used so that earlier years were
used for training and later years for evaluation.

### Model Performance

| Metric | Value |
|---|---:|
| MAE | 85.87 |
| RMSE | 142.96 |
| R² Score | 0.787 |

The R² score indicates that approximately 78.7% of the variation in
the test target was explained by the model.

---

## Technology Stack

### Programming & Analysis
- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook

### Visualization
- Matplotlib
- Streamlit

### Machine Learning
- Random Forest Regressor
- Joblib

### Application
- Streamlit

---

## Project Structure

```text
crime_against_women/
│
├── app.py
├── crime_against_women_analysis.ipynb
│
├── up_women_crime.csv
├── district_risk_profile.csv
│
├── crime_prediction_model.pkl
├── model_performance.csv
├── prediction_comparison.csv
├── feature_importance.csv
│
├── requirements.txt
├── .gitignore
└── README.md