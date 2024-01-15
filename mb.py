import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# Load the data
df = pd.read_csv("Employee.csv")

# Preprocessing
df.columns = df.columns.str.lower()
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
df['everbenched'] = df['everbenched'].map({'Yes': 1, 'No': 0})
df['education'] = df['education'].map({'Bachelors': 0, 'Masters': 1, 'PHD': 2})
df = pd.get_dummies(df, columns=['city'])

scaler = MinMaxScaler()
scaler.fit(df[['joiningyear', 'paymenttier', 'experienceincurrentdomain']])
df[['joiningyear', 'paymenttier', 'experienceincurrentdomain']] = scaler.transform(
    df[['joiningyear', 'paymenttier', 'experienceincurrentdomain']]
)

# Split the data into features and labels
features = df.drop(['leaveornot'], axis=1)
labels = df['leaveornot']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create a RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Streamlit App
def main():
    st
