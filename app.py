import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import joblib  # Assuming your model is saved using joblib


model = joblib.load('grid_search.joblib')
encoder_dict = joblib.load('encoder.joblib')

# Define columns
cols = ['Education', 'JoiningYear', 'City', 'PaymentTier', 'Age', 'Gender', 'EverBenched', 'ExperienceInCurrentDomain']

def main():
    st.title('FUTURE EMPLOYEE ATTRITION')
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Income Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # User input form
    education = st.selectbox('Education:', ['Bachelor', 'Master'])
    joining_year = st.slider('Joining Year:', 2012, 2018, 2015)
    city = st.selectbox('City:', ['Bangalore', 'Pune', 'New Delhi'])
    payment_tier = st.selectbox('Payment Tier:', [1, 2, 3])
    age = st.text_input("Age", "0")
    gender = st.selectbox('Gender:', ['Male', 'Female'])
    ever_benched = st.selectbox('Ever Benched:', ['Yes', 'No'])
    experience_in_current_domain = st.number_input('Experience in Current Domain:')

    user_input_form = st.form('user_input_form')
    additional_inputs = ['AdditionalInput1', 'AdditionalInput2']
    for input_name in additional_inputs:
        setattr(user_input_form, input_name, user_input_form.text_input(input_name, f"Enter {input_name}"))

    # Submit button
   if user_input_form.form_submit_button('Predict'):
       
        # Create DataFrame from user input
        user_data = {key: [getattr(user_input_form, key)] for key in additional_inputs}
        user_df = pd.DataFrame(user_data)

        # One-hot encode categorical columns
        categorical_cols = ['Education', 'City', 'Gender', 'EverBenched']
        user_df = pd.get_dummies(user_df, columns=categorical_cols)

        # Ensure that the columns in user_df match the columns used during model training
        expected_columns = ['Education_Bachelor', 'Education_Master', 'JoiningYear', 'City_Bangalore', 'City_Pune', 'City_New Delhi',
                            'PaymentTier', 'Age', 'Gender_Female', 'Gender_Male', 'EverBenched_No', 'EverBenched_Yes',
                            'ExperienceInCurrentDomain']
        
        # Align columns in user_df
        user_df = user_df.reindex(columns=expected_columns, fill_value=0)

        # Make predictions using the loaded model
        prediction = model.predict(user_df)

        # Display prediction
        st.subheader('Prediction:')
        st.write('Leave: Yes' if prediction[0] == 1 else 'Leave: No')
