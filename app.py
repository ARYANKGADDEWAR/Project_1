import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# Set the path to the current directory where the script is located
# This helps in finding the model and data files when deployed
SCRIPT_DIR = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()

# 1. Load the trained model
model_filename = os.path.join(SCRIPT_DIR, 'random_forest_regressor_model.joblib')
try:
    rf_reg_model = joblib.load(model_filename)
    st.success(f"Successfully loaded the model: {model_filename}")
except FileNotFoundError:
    st.error(f"Error: The model file '{model_filename}' was not found. Please ensure it is in the same directory as the Streamlit app.")
    st.stop() # Stop the app if the model isn't found
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# 2. Reload the original data to fit LabelEncoders consistently
data_filename = os.path.join(SCRIPT_DIR, 'Salary_Data.csv')
try:
    df_original = pd.read_csv(data_filename)
    # Fill NaN values with mode, consistent with preprocessing steps during training
    df_original.fillna(df_original.mode().iloc[0], inplace=True)
    st.success(f"Successfully loaded original data for LabelEncoders from: {data_filename}")
except FileNotFoundError:
    st.error(f"Error: 'Salary_Data.csv' not found at '{data_filename}'.")
    st.error("This file is necessary to correctly re-initialize the LabelEncoders. Please ensure it is in the same directory as the Streamlit app.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading or processing 'Salary_Data.csv': {e}")
    st.stop()

# Initialize and fit LabelEncoders for categorical columns
label_encoders = {}
categorical_cols = ['Gender', 'Education Level', 'Job Title']
for col in categorical_cols:
    le = LabelEncoder()
    # Fit on all unique values from the original (filled) dataset
    le.fit(df_original[col].astype(str).unique()) # Convert to string to handle potential mixed types if any
    label_encoders[col] = le

# 3. Define the prediction function
def predict_salary(age, gender, education_level, job_title, years_of_experience):
    try:
        gender_encoded = label_encoders['Gender'].transform([gender])[0]
        education_level_encoded = label_encoders['Education Level'].transform([education_level])[0]
        job_title_encoded = label_encoders['Job Title'].transform([job_title])[0]
    except ValueError as e:
        return f"Error encoding categorical input: {e}. Please ensure inputs are valid options."

    input_df = pd.DataFrame([[age, gender_encoded, education_level_encoded, job_title_encoded, years_of_experience]],
                            columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])

    prediction = rf_reg_model.predict(input_df)[0]
    return prediction

# 4. Create Streamlit Interface
st.title("Salary Prediction App")
st.write("Enter employee details to predict their salary using a Random Forest Regressor model.")

# Get unique values for dropdowns/radio buttons from the fitted LabelEncoders' classes_
gender_options = label_encoders['Gender'].classes_.tolist()
education_options = label_encoders['Education Level'].classes_.tolist()
job_title_options = label_encoders['Job Title'].classes_.tolist()

with st.form("salary_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.radio("Gender", gender_options)
    education_level = st.selectbox("Education Level", education_options)
    job_title = st.selectbox("Job Title", job_title_options)
    years_of_experience = st.number_input("Years of Experience", min_value=0, max_value=60, value=5)

    submitted = st.form_submit_button("Predict Salary")

    if submitted:
        salary = predict_salary(age, gender, education_level, job_title, years_of_experience)
        if isinstance(salary, str):
            st.error(salary) # Display error if prediction failed due to encoding issue
        else:
            st.success(f"Predicted Salary: ${salary:,.2f}")

# Instructions for deployment
st.markdown("""
---
### How to deploy this application:
1. Save this code as `app.py` in a directory.
2. Ensure `random_forest_regressor_model.joblib` and `Salary_Data.csv` are in the same directory.
3. Open your terminal, navigate to that directory, and run: `streamlit run app.py`
""")
