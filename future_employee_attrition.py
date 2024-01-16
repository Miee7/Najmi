# -*- coding: utf-8 -*-
"""Future employee attrition.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VI3JjHUzQbv8LFQSLrJuXq-Q-xHb2DoV
"""

# Commented out IPython magic to ensure Python compatibility.
# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
# %matplotlib inline

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn import metrics



df= pd.read_csv("Employee.csv")
df.head()

df.shape

df.info()

df.describe()

# prompt: make all column lower case

df.columns = df.columns.str.lower()

# prompt: convert gender and everbenched into integer

df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
df['everbenched'] = df['everbenched'].map({'Yes': 1, 'No': 0})

df.head()

leaveornot_counts = df['leaveornot'].value_counts()

plt.bar(leaveornot_counts.index, leaveornot_counts.values)

# Adding labels to the bars
for i, count in enumerate(leaveornot_counts.values):
    plt.text(i, count + 10, str(count), ha='center', va='bottom')

plt.xticks(leaveornot_counts.index, [str(leaveornot) for leaveornot in leaveornot_counts.index])
plt.xlabel('Leave or Not')
plt.ylabel('Count')
plt.title('Leave or Not Distribution')

# Adding labels '0' and '1' to the upper right
plt.text(0.95, 0.95, '0: Not Leave\n1: Leave', transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.show()

"""### **Bar Chart for PaymentTier**"""

# 1. Bar Chart for PaymentTier
plt.figure(figsize=(8, 6))
df['paymenttier'].value_counts().plot(kind='bar',color=['red','blue','green'])
plt.title('Distribution of Employees Across Payment Tiers')
plt.xlabel('Payment Tier')
plt.ylabel('Number of Employees')

plt.show()

"""1.seramai 3500 orang mempunyai gaji rendahb40
kedua gaji sederhana m40
t20
"""

plt.figure(figsize=(8, 6))
plt.hist(df['age'], bins=20, color='purple', edgecolor='black')
plt.title('Distribution of Employee Ages')
plt.xlabel('Age')
plt.ylabel('Number of Employees')
plt.show()

"""DALAM GOLONGAN UMUR 24 HINGGA 28 IALAH PEKERJA YANG BEKERJA DI SEBUAH SYARIKAT"""

import matplotlib.pyplot as plt

# Assuming df is your DataFrame
plt.figure(figsize=(8, 8))

# Plotting the pie chart with custom labels
df['gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'lightcoral'],
                                 labels=['Male', 'Female'])

# Adding title
plt.title('Gender Distribution')

# Display the plot
plt.show()

# 6. Box Plot for LeaveOrNot and ExperienceInCurrentDomain
plt.figure(figsize=(8, 6))
sns.boxplot(x='leaveornot', y='experienceincurrentdomain', data=df)
plt.title('Box Plot of Experience in Current Domain for Employees Who Left or Not')
plt.xlabel('Leave or Not')
plt.ylabel('Experience in Current Domain')
plt.show()

education_over_years = df.groupby('joiningyear')['education'].value_counts().unstack()
education_over_years.plot(kind='line', marker='o', figsize=(10, 6))
plt.title('Number of Employees with Different Education Levels Over Years')
plt.xlabel('Joining Year')
plt.ylabel('Count')
plt.show()

# prompt: generate heatmap

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='Blues')
plt.title('Correlation Heatmap of Employee Features')
plt.show()

df['education'].unique()

df['education'] = df['education'].map({'Bachelors': 0, 'Masters': 1, 'PHD': 2})

# prompt: one hot encode column city

from sklearn.preprocessing import OneHotEncoder
df = pd.get_dummies(df, columns=['city'])

# prompt: do min max scalling to clumn joiningYear,PaymentTier,ExperienceInCurrentDomain

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler to the training data
scaler.fit(df[['joiningyear', 'paymenttier', 'experienceincurrentdomain']])

# Transform the training data
df[['joiningyear', 'paymenttier', 'experienceincurrentdomain']] = scaler.transform(df[['joiningyear', 'paymenttier', 'experienceincurrentdomain']])

df.sample(5)

df.info()

X = df.drop(['leaveornot'], axis=1)
y = df['leaveornot']

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_rows', None)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training set
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)

# Print predicted and actual values side by side
result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nActual vs Predicted:")
print(result_df)

from sklearn.model_selection import GridSearchCV

# Define the hyperparameter grid to search
param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Make predictions on the testing set using the best model
y_pred = best_model.predict(X_test)

# Evaluate the best model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Best Hyperparameters: {best_params}")
print(f"Best Model Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Split the data into features and labels
features = df.drop(['leaveornot'], axis=1)  # Assuming you drop the target column
labels = df['leaveornot']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create a RandomForestClassifier (you can choose a different classifier based on your needs)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

# Display the results
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:\n', classification_report_str)

# prompt: save gridsearchcv into joblib

import joblib
joblib.dump(grid_search, 'grid_search.joblib')