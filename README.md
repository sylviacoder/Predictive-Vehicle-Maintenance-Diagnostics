# Predictive-Vehicle-Maintenance-Diagnostics
An end-to-end machine learning project focused on predicting engine health conditions using sensor telemetry data. The system analyzes engine operational parameters such as RPM, pressure, and temperature readings to determine whether an engine is operating normally or showing signs of failure.
The project combines predictive maintenance concepts with machine learning to demonstrate how sensor data can be used for early fault detection in industrial and automotive systems.

## Project Overview

Modern vehicles and industrial engines generate continuous streams of sensor data during operation. Monitoring these readings manually is difficult, especially at scale.
This project aims to automate engine condition monitoring by building a machine learning model capable of classifying engine states as either:

**Healthy Engine**
**Faulty Engine**

The solution can help support:
predictive maintenance
early anomaly detection
reduced downtime
improved operational safety
Problem Statement

Traditional maintenance approaches often rely on:
fixed servicing schedules
reactive repairs after breakdowns occur

These methods can lead to:
unnecessary maintenance costs
unexpected equipment failure
operational inefficiency

This project explores how machine learning can be used to identify hidden patterns in engine telemetry data and predict engine conditions before severe failures occur.

Dataset Information
The dataset contains over 19,000 engine sensor observations with numerical telemetry readings collected from engine systems.
Engine rpm	Engine rotational speed
Lub oil pressure	Lubrication oil pressure
Fuel pressure	Fuel system pressure
Coolant pressure	Cooling system pressure
lub oil temp	Lubrication oil temperature
Coolant temp	Cooling system temperature
Target Variable
Label	Meaning
1	Healthy Engine
0	Faulty Engine
Feature Engineering

A temperature differential feature was introduced to capture thermal imbalance within the engine system.
```
df['Temp_Difference'] = (
    df['lub oil temp'] - df['Coolant temp']
)
```

This helps the model better understand:
overheating patterns
cooling inefficiencies
thermal stress behavior
Exploratory Data Analysis

Several visualizations were performed to better understand the dataset, including:
target distribution analysis
correlation heatmaps
feature distributions
boxplots by engine condition
feature importance analysis

The analysis showed that variables such as:
engine RPM
lubrication oil pressure
fuel pressure
temperature readings
had strong relationships with engine health conditions.

## Machine Learning Workflow

The project follows a complete machine learning pipeline:

1. Data Preprocessing
feature engineering
train-test splitting
feature scaling using StandardScaler

3. Model Training
Multiple models were explored, including:
Logistic Regression
Random Forest Classifier

3. Model Evaluation

Models were evaluated using:
Accuracy
Precision
Recall
F1-score
Cross-validation
Confusion Matrix
Model Performance

The Random Forest model produced the strongest overall performance due to its ability to capture nonlinear relationships between sensor readings.
The model successfully identified:
healthy engine behavior
abnormal operational patterns
fault-related sensor combinations
Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Joblib
Streamlit

## Streamlit Deployment
The project includes an interactive Streamlit application where users can:
enter engine sensor readings
predict engine health status
receive real-time engine condition feedback

The deployment demonstrates how machine learning models can be integrated into monitoring systems for operational decision-making.

Example Use Cases
This type of system can be applied in:
fleet management
industrial equipment monitoring
manufacturing plants
automotive diagnostics
smart maintenance systems

## Key Insights
The analysis revealed that:
abnormal RPM patterns may indicate engine stress
unstable pressure readings correlate with potential faults
thermal imbalance can be an early sign of engine degradation

These insights align with real-world predictive maintenance practices used in industrial monitoring systems.

## Future Improvements
Potential improvements include:
real-time IoT sensor integration
deep learning models for time-series prediction
anomaly detection systems
Remaining Useful Life (RUL) estimation
cloud deployment
SHAP explainability integration
live monitoring dashboards
Skills Demonstrated

This project demonstrates practical experience in:
predictive maintenance analytics
machine learning classification
feature engineering
data preprocessing
exploratory data analysis
model evaluation
deployment with Streamlit
industrial AI concepts

## Conclusion
This project demonstrates how machine learning can be used to transform raw engine telemetry data into actionable maintenance insights.
By combining sensor analytics with predictive modeling, the system provides a foundation for intelligent engine monitoring and early fault detection systems commonly used in modern industrial and automotive environments.


