# Diabetes-Prediction-using-Logistic-Regression---Healthcare-ML


Logistic Regression - Diabetes Prediction
This project implements a Logistic Regression model to predict diabetes based on patient health metrics. The model is trained on the Pima Indians Diabetes Dataset and evaluated using various performance metrics.

📁 Project Structure
logisticREG.ipynb: Main Jupyter notebook containing the complete implementation

diabetes.csv: Dataset used for training and testing

README.md: Project documentation (this file)

🎯 Objective
Build a machine learning model to predict whether a patient has diabetes based on diagnostic measurements, using Logistic Regression.

📊 Dataset
The project uses the Pima Indians Diabetes Dataset which contains:

768 patient records

8 medical predictor variables

1 target variable (Outcome: 0 = No Diabetes, 1 = Diabetes)

Features:
Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

DiabetesPedigreeFunction

Age

🛠️ Implementation Steps
Data Loading & Exploration

Import necessary libraries (NumPy, Pandas, Matplotlib, Seaborn)

Load and inspect the dataset

Check for missing values

Data Preprocessing

Split data into features (X) and target (y)

Split into training and testing sets (75-25 split)

Feature scaling using StandardScaler

Model Training

Implement Logistic Regression with random_state=0

Train the model on scaled training data

Model Evaluation

Make predictions on test set

Calculate accuracy score

Generate confusion matrix

Visualize results

📈 Results
The model achieves:

Accuracy: 80% (original implementation)

Accuracy: 75% (AI-assisted implementation)

Confusion Matrix (Original Implementation):
text
[[118  12]
 [ 26  36]]
🚀 How to Run
Ensure you have Python and Jupyter Notebook installed

Install required packages:

bash
pip install numpy pandas matplotlib seaborn scikit-learn
Download the diabetes.csv dataset

Open and run logisticREG.ipynb in Jupyter Notebook

📋 Dependencies
Python 3.x

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

🔍 Key Insights
The dataset contains no missing values

Feature scaling significantly improves model performance

Logistic Regression proves effective for this binary classification problem

The model shows good predictive capability for diabetes diagnosis

📝 Notes
The project includes both manual implementation and AI-assisted approaches

Different train-test splits (25% vs 20%) were tested

Comprehensive visualization of results is provided
