
import streamlit as st
import joblib
import numpy as np

# Define the path to your model
model_path = '/content/random_forest_regressor_model.joblib'

# Load the model
@st.cache_resource
def load_model():
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_path}")
        return None

model = load_model()

st.title("Random Forest Regressor Model 1 Front-end")
st.write("Enter feature values to get a prediction from your RandomForestRegressor model.")

# Define a prediction function
def predict_output(model, feature1, feature2, feature3, feature4, feature5):
    if model is None:
        return "Model not loaded. Cannot make prediction."

    features = np.array([[feature1, feature2, feature3, feature4, feature5]])

    try:
        prediction = model.predict(features)
        return f"Predicted Value: {prediction[0]:.2f}"
    except Exception as e:
        return f"Error during prediction: {e}"

# Create input fields in Streamlit
st.sidebar.header("Input Features")
feature1 = st.sidebar.number_input("Age ", value=30.0)
feature2 = st.sidebar.number_input("Gender ", value=1.0) # Assuming 1 for Male, 0 for Female, or similar encoding
feature3 = st.sidebar.number_input("Education Level ", value=12.0)
feature4 = st.sidebar.number_input("Job title ", value=5.0) # Assuming some numerical encoding for job title
feature5 = st.sidebar.number_input("Years OF Experience ", value=5.0)

if st.sidebar.button("Predict"):
    if model is not None:
        result = predict_output(model, feature1, feature2, feature3, feature4, feature5)
        st.success(result)
    else:
        st.error("Model could not be loaded. Please check the model path.")
