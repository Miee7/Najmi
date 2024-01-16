import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import joblib  # Assuming your model is saved using joblib
import streamlit_aggrid as ag

def main():
    # Load your predictive model
    # Replace 'your_model_file.joblib' with the actual filename of your saved model
    model = joblib.load('grid_search.joblib')

    # Streamlit app
    st.title('FUTURE EMPLOYEE ATTRITION')

    # User input form
    st.sidebar.header('User Input:')

    input_options = {
        'Education': st.selectbox('Education:', ['Bachelor', 'Master']),
        'JoiningYear': st.slider('Joining Year:', 2012, 2018, 2015),
        'City': st.selectbox('City:', ['Bangalore', 'Pune', 'New Delhi']),
        'PaymentTier': st.selectbox('Payment Tier:', [1, 2, 3]),
        'Age': st.number_input('Age:'),
        'Gender': st.selectbox('Gender:', ['Male', 'Female']),
        'EverBenched': st.selectbox('Ever Benched:', ['Yes', 'No']),
        'ExperienceInCurrentDomain' : st.number_input('Experience in Current Domain:')
    }
        
   # Display user input form
    user_input_form = st.form('user_input_form')
    for key, value in input_options.items():
        setattr(user_input_form, key, user_input_form.text_input(key, value))

    # Submit button
    submitted = user_input_form.form_submit_button('Predict')

    if submitted:
        # Create DataFrame from user input
        user_data = {key: [getattr(user_input_form, key)] for key in input_options.keys()}
        user_df = pd.DataFrame(user_data)

        # One-hot encode 'JobRole', 'Department', 'BusinessTravel', and 'EducationField' columns
        encoder = OneHotEncoder(sparse=False)
        encoded_cols = ['City']
        user_df[encoded_cols] = encoder.fit_transform(user_df[encoded_cols])

        # Min-Max scaling for selected columns
        columns_to_scale = ['JoiningYear', 'PaymentTier', 'ExperienceInCurrentDomain']
        scaler = MinMaxScaler()

        # Fit the scaler to the user input data
        scaler.fit(user_df[columns_to_scale])

        # Transform the user input data
        user_df[columns_to_scale] = scaler.transform(user_df[columns_to_scale])

        # Ensure that the columns in user_df match the columns used during model training
        expected_columns = ['Age', 'City_Bangalore', 'City_Pune', 'City_New Delhi', 'Education_Bachelor', 'Education_Master',
                        'EverBenched_No', 'EverBenched_Yes', 'ExperienceInCurrentDomain', 'JoiningYear', 'PaymentTier']

        # Align columns in user_df
        user_df = user_df.reindex(columns=expected_columns, fill_value=0)

        # Make predictions using the loaded model
        prediction = model.predict(user_df)

        # Display prediction
        st.subheader('Prediction:')
        st.write('Leave: Yes' if prediction[0] == 1 else 'Leave: No')

if __name__ == '__main__':
    main()
