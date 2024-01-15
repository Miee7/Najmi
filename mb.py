# Import necessary libraries
import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load employee data
data = pd.read_csv('employee_data.csv')

# Define the columns that will be used as features
features = ['job_role', 'department', 'business_travel', 'education_field', 'marital_status', 'monthly_income', 'total_working_years', 'years_at_company', 'years_in_current_role', 'age']

# Define the target column
target = 'left_company'

# Split the data into features and target
X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model on the training data
model = LogisticRegression()
model.fit(X_train, y_train)

# Define a function to predict attrition risk
def predict_attrition(job_role, department, business_travel, education_field, marital_status, monthly_income, total_working_years, years_at_company, years_in_current_role, age):
    input_features = pd.DataFrame([[job_role, department, business_travel, education_field, marital_status, monthly_income, total_working_years, years_at_company, years_in_current_role, age]], columns=features)
    prediction = model.predict(input_features)
    return prediction[0]

# Define the Streamlit app
def main():
    # Set up the app title and image
    st.title('Employee Attrition Predictor')
    image = Image.open('employee.png')
    st.image(image, width=300)

    # Get user input for each feature
    job_role = st.selectbox('Job Role', ['Sales', 'Marketing', 'Healthcare Rep', 'Human Resources'])
    department = st.selectbox('Department', ['Sales', 'Marketing', 'Healthcare', 'Human Resources'])
    business_travel = st.selectbox('Business Travel', ['Travel', 'Non-Travel'])
    education_field = st.selectbox('Education Field', ['Sales', 'Marketing', 'Healthcare', 'Human Resources'])
    marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
    monthly_income = st.number_input('Monthly Income', min_value=0.0, step=100.0)
    total_working_years = st.number_input('Total Working Years', min_value=0.0, step=1.0)
    years_at_company = st.number_input('Years at Company', min_value=0.0, step=1.0)
    years_in_current_role = st.number_input('Years in Current Role', min_value=0.0, step=1.0)
    age = st.number_input('Age', min_value=0, step=1)

    # Display the user input
    st.write('')
    st.write('**Employee Information:**')
    st.write('')
    st.write('Job Role:', job_role)
    st.write('Department:', department)
    st.write('Business Travel:', business_travel)
    st.write('Education Field:', education_field)
    st.write('Marital Status:', marital_status)
    st.write('Monthly Income:', monthly_income)
    st.write('Total Working Years:', total_working_years)
    st.write('Years at Company:', years_at_company)
    st.write('Years in Current Role:', years_in_current_role)
    st.write('Age:', age)

    # Predict attrition risk
    if st.button('Predict'):
        prediction = predict_attrition(job_role, department, business_travel, education_field, marital_status, monthly_income, total_working_years, years_at_company, years_in_current_role, age)
        if prediction == 0:
            st.write('**Attrition Risk:** Low')
        else:
            st.write('**Attrition Risk:** High')

# Run the Streamlit app
if __name__ == '__main__':
    main()
