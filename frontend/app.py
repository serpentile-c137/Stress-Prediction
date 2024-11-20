import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go  # For the dial meter

# Load the model
with open(r'C:\Users\sahus\HyperFlux\xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize the scaler (assuming the model was trained with scaled data)
scaler = StandardScaler()

# Set up page configuration
st.set_page_config(page_title="XGBoost Prediction App", page_icon="🔍", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 36px;
        color: #4CAF50;
    }
    .sub-title {
        text-align: center;
        font-size: 18px;
        color: gray;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.markdown("<h1 class='main-title'>XGBoost Classifier Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Predict the cluster based on HR, EDA, TEMP, and BVP values</p>", unsafe_allow_html=True)
st.write("### Enter feature values below and click **Predict** to see the result:")

# Input fields for each feature
feature_1 = st.number_input("Enter EDA (Electrodermal Activity)", min_value=-50.0, max_value=100.0, value=0.42)
feature_2 = st.number_input("Enter TEMP (Temperature)", min_value=0.0, max_value=150.0, value=33.3)
feature_3 = st.number_input("Enter BVP (Blood Volume Pulse)", min_value=-60.0, max_value=100.0, value=-18.0)
feature_4 = st.number_input("Enter HR (Heart Rate)", min_value=0.0, max_value=200.0, value=177.0)

# Collect all feature values in a DataFrame for scaling
user_input = pd.DataFrame([[feature_1, feature_2, feature_3, feature_4]], columns=['EDA', 'TEMP', 'BVP','HR'])

# Display input data in the main page
st.write("### Selected Input Values:")
st.dataframe(user_input)

# Scale the input data
user_input_reshape = user_input.to_numpy().reshape(-1, 1)
user_input_scaled = scaler.fit_transform(user_input_reshape)
single_row_array = user_input_scaled.reshape(1, -1)

# Button to make a prediction
if st.button("Predict"):
    # Predict the cluster
    prediction = model.predict(single_row_array)
    probability = model.predict_proba(single_row_array)
    stress = round(probability[0][1] * 100, 2)  # Probability of stress level

    # Display the result with a success message
    st.markdown("<h2 style='color: #4CAF50;'>Prediction Result:</h2>", unsafe_allow_html=True)

    # Create a dial meter for the stress level using Plotly
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=stress,
        title={'text': "Stress Level (%)"},
        gauge={
            'axis': {'range': [0, 100]},
        'steps': [
            {'range': [0, 25], 'color': "lightgray"},
            {'range': [25, 50], 'color': "gray"},
            {'range': [50, 75], 'color': "red"},
            {'range': [75, 100], 'color': "blue"}
        ],

        'threshold': {
        'line': {'color': "red", 'width': 4},
        'thickness': 0.75,
        'value': 490}
        }
    ))

    # Display the stress meter
    st.title('Stress-O-meter')
    st.plotly_chart(fig)

#st.write(f"#### The predicted cluster is: **{int(prediction[0])}**")
    #st.write(f"#### The probability of the prediction: **{stress}**%")
    if stress:
        try:
            if stress > 0.0 and stress <= 25.0:
                st.write(f"** You have {stress} % stress predicted. **")
                st.write("Stress level is low. Try to maintain your stress level for good health")
            elif stress > 25.0 and stress <= 50.0:
                st.write(f"You have {stress} % stress predicted.")
                st.write("Stress level is mild.Try to doing exercises to reduce your stress level for good health.")
            elif stress > 50.0 and stress <= 75.0:
                st.write(f"You have {stress} % stress predicted.")
                st.write("Stress level is moderate. Consider seeking help from a healthcare professional.")
            elif stress > 75.0 and stress <= 100.0:
                st.write(f"**You have {stress} % stress predicted.**")
                st.write("**Stress level is high. Please consult with a healthcare professional Immediately.**")
        except ValueError:
            st.error("Please enter a valid number.")
    else:
        st.write("Please enter a number above.")

    #st.success("Prediction successful! 🎉")

# Footer with contact info
st.markdown("<hr>", unsafe_allow_html=True)
st.write("Made with ❤️ by [Your Name](https://github.com/YourProfile)")
