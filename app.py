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
    Education = st.selectbox('education:', ['Bachelor', 'Master'])
    JoiningYear = st.slider('joiningyear:', 2012, 2018, 2015)
    City = st.selectbox('city:', ['Bangalore', 'Pune', 'New Delhi'])
    PaymentTier = st.selectbox('Payment Tier:', [1, 2, 3])
    Age = st.text_input("age", "0")
    Gender = st.selectbox('gender:', ['Male', 'Female'])
    EverBenched = st.selectbox('everbenched:', ['Yes', 'No'])
    ExperienceInCurrentDomain = st.number_input('experienceincurrentdomain:')

    user_input_form = st.form('user_input_form')

    # Submit button
    if user_input_form.form_submit_button('Predict'):
        # Create DataFrame from user input
        user_data = {
            'Education': [education],
            'JoiningYear': [joiningyear],
            'City': [city],
            'PaymentTier': [paymenttier],
            'Age': [age],
            'Gender': [gender],
            'EverBenched': [everbenched],
            'ExperienceInCurrentDomain': [experienceincurrentdomain],
    }

        user_df = pd.DataFrame(user_data)

        # One-hot encode categorical columns
        categorical_cols = ['Education', 'City', 'Gender', 'EverBenched']
        user_df = pd.get_dummies(user_df, columns=categorical_cols)

        # Ensure that the columns in user_df match the columns used during model training
        expected_columns = ['education', 'joiningyear', 'city_Bangalore', 'city_New Delhi', 'city_Pune', 'paymenttier', 'age', 'gender', 'everbenched',
                'experienceincurrentdomain']
        
        # Align columns in user_df
        user_df = user_df.reindex(columns=expected_columns, fill_value=0)

        # Make predictions using the loaded model
        prediction = model.predict(user_df)

        # Display prediction
        st.subheader('Prediction:')
        st.write('Leave: Yes' if prediction[0] == 1 else 'Leave: No')
        
if _name_ == '_main_':
    main()
