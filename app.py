import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import joblib  # Assuming your model is saved using joblib

def main():
    # Load your predictive model
    # Replace 'your_model_file.joblib' with the actual filename of your saved model
    model = joblib.load('grid_search.joblib')

    # Streamlit app
    st.title('FUTURE EMPLOYEE ATTRITION')

    # User input form
    st.sidebar.header('User Input:')
    education = st.sidebar.selectbox('Education:', ['Bachelor', 'Master'])
    joining_year = st.sidebar.slider('Joining Year:', 2012, 2018, 2015)
    city = st.sidebar.selectbox('City:', ['Bangalore', 'Pune', 'New Delhi'])
    payment_tier = st.sidebar.selectbox('Payment Tier:', [1, 2, 3])
    age = st.sidebar.number_input('Age:')
    gender = st.sidebar.selectbox('Gender:', ['Male', 'Female'])
    ever_benched = st.sidebar.selectbox('Ever Benched:', ['Yes', 'No'])
    experience_in_current_domain = st.sidebar.number_input('Experience in Current Domain:')

    # Create DataFrame from user input
    user_data = {
        'Education': [education],
        'JoiningYear': [joining_year],
        'City': [city],
        'PaymentTier': [payment_tier],
        'Age': [age],
        'Gender': [gender],
        'EverBenched': [ever_benched],
        'ExperienceInCurrentDomain': [experience_in_current_domain]
    }

    user_df = pd.DataFrame(user_data)

    # Display user input DataFrame
    st.subheader('User Input DataFrame')
    st.write(user_df)

    # One-hot encode 'City' column
    user_df = pd.get_dummies(user_df, columns=['City'])

    # Display DataFrame after one-hot encoding
    st.subheader('DataFrame after One-Hot Encoding')
    st.write(user_df)

    # Min-Max scaling for selected columns
    columns_to_scale = ['JoiningYear', 'PaymentTier', 'ExperienceInCurrentDomain']
    scaler = MinMaxScaler()

    # Fit the scaler to the user input data
    scaler.fit(user_df[columns_to_scale])

    # Transform the user input data
    user_df[columns_to_scale] = scaler.transform(user_df[columns_to_scale])

    # Display DataFrame after Min-Max scaling
    st.subheader('DataFrame after Min-Max Scaling')
    st.write(user_df)

    # Ensure that the columns in user_df match the columns used during model training
    expected_columns = ['Age', 'City_Bangalore', 'City_Pune', 'City_New Delhi', 'Education_Bachelor', 'Education_Master',
                    'EverBenched_No', 'EverBenched_Yes', 'ExperienceInCurrentDomain', 'JoiningYear', 'PaymentTier']

    # Print and check columns in user_df
    print("Columns in user_df:", user_df.columns)

    # Align columns in user_df
    user_df = user_df.reindex(columns=expected_columns, fill_value=0)

    # Print columns after alignment
    print("Columns in user_df after alignment:", user_df.columns)

    # Make predictions using the loaded model
    prediction = model.predict(user_df)

    # Display prediction
    st.subheader('Prediction:')
    st.write('Leave: Yes' if prediction[0] == 1 else 'Leave: No')

if __name__ == '__main__':
    main()
