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
    <h2 style="color:white;text-align:center;">FUTURE EMPLOYEE ATTRITION PRIDICTION</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # User input form
    education = st.selectbox('Education:', ['Bachelor', 'Master'])
    joiningyear = st.slider('JoiningYear:', 2012, 2018, 2015)
    city = st.selectbox('City:', ['Bangalore', 'Pune', 'New Delhi'])
    paymenttier = st.selectbox('Payment Tier:', [1, 2, 3])
    age = st.text_input("Age", "0")
    gender = st.selectbox('Gender:', ['Male', 'Female'])
    everbenched = st.selectbox('EverBenched:', ['Yes', 'No'])
    experienceincurrentdomain = st.number_input('ExperienceInCurrentDomain:')

    user_input_form = st.form('user_input_form')

    # Submit button
    if user_input_form.form_submit_button('Predict'):
        # Create DataFrame from user input
        user_data = {
            'Education': [Education],
            'JoiningYear': [JoiningYear],
            'City': [City],
            'PaymentTier': [PaymentTier],
            'Age': [Age],
            'Gender': [Gender],
            'EverBenched': [EverBenched],
            'ExperienceInCurrentDomain': [ExperienceInCurrentDomain],
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
        
if __name__=='__main__': 
    main()
