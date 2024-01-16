import streamlit as st
import joblib

classifier = joblib.load('grid_search.joblib')

st.title("Employee Leave Prediction App")
