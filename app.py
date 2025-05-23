import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('random_forest_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Brain Tumor Prediction App")

# Create input fields
age = st.number_input("Age", min_value=0, max_value=120)
gender = st.selectbox("Gender", ["Male", "Female"])
tumor_type = st.text_input("Tumor Type")
tumor_size = st.number_input("Tumor Size (mm)", min_value=0.0)
location = st.text_input("Tumor Location")
histology = st.text_input("Histology")
stage = st.selectbox("Stage", ["I", "II", "III", "IV"])
symptom_1 = st.text_input("Symptom 1")
symptom_2 = st.text_input("Symptom 2")
symptom_3 = st.text_input("Symptom 3")
radiation = st.selectbox("Radiation Treatment", ["Yes", "No"])
surgery = st.selectbox("Surgery Performed", ["Yes", "No"])
chemotherapy = st.selectbox("Chemotherapy", ["Yes", "No"])
family_history = st.selectbox("Family History", ["Yes", "No"])
mri_result = st.text_input("MRI Result")

# Assume the model predicts Follow_Up_Required
if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Tumor_Type": tumor_type,
        "Tumor_Size": tumor_size,
        "Location": location,
        "Histology": histology,
        "Stage": stage,
        "Symptom_1": symptom_1,
        "Symptom_2": symptom_2,
        "Symptom_3": symptom_3,
        "Radiation_Treatment": radiation,
        "Surgery_Performed": surgery,
        "Chemotherapy": chemotherapy,
        "Family_History": family_history,
        "MRI_Result": mri_result
    }])
    
    # You might need to preprocess this input_df to match training format
    prediction = model.predict(input_df)
    st.success(f"Prediction: {prediction[0]}")
