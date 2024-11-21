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
    /* Dark theme background with soft color palette */
    .stApp {
        background-color: #2a2a2a;  /* Dark Gray */
        color: #e0e0e0;  /* Light Gray Text */
    }
    
    /* Title and subtitle styling */
    .main-title {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        color: #00cc66;  /* Vibrant green */
        margin-top: -10px;
        text-shadow: 2px 2px 5px rgba(0, 204, 102, 0.5);
    }

    .sub-title {
        text-align: center;
        font-size: 1.2em;
        color: #cccccc;  /* Lighter gray */
        font-style: italic;
        margin-bottom: 20px;
    }

    /* Input fields styling */
    .stNumberInput input {
        background-color: #444444;  /* Darker background for inputs */
        color: #ffffff;  /* White text */
        border-radius: 8px;
        padding: 10px;
        font-size: 1em;
        border: 1px solid #666666;  /* Border color */
    }

    /* Focus effect for input fields */
    .stNumberInput input:focus {
        border: 1px solid #00cc66;  /* Green border on focus */
        box-shadow: 0 0 5px #00cc66;  /* Green glow on focus */
    }

    /* Button styling with hover effect */
    .stButton>button {
        background-color: #4CAF50;  /* Green */
        color: #ffffff;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        box-shadow: 0px 4px 10px rgba(76, 175, 80, 0.4);
        transition: 0.3s ease;
    }
    
    /* Button hover effect */
    .stButton>button:hover {
        background-color: #43a047;  /* Darker green on hover */
        transform: translateY(-2px);
        box-shadow: 0px 6px 12px rgba(67, 160, 71, 0.6);
    }

    /* Text styling for st.write */
    .stMarkdown, .stWrite, .stText {
        color: #e0e0e0;  /* Light Gray Text for all st.write() */
    }

    /* Footer styling */
    hr {
        border: none;
        border-top: 1px solid #444444;
        margin-top: 30px;
        margin-bottom: 10px;
    }

    .footer {
        text-align: center;
        font-size: 0.9em;
        color: #999999;
    }

    /* Add subtle animations */
    .fade-in {
        animation: fadeIn 1s ease-in-out;
    }
    
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)


# Title and description
st.markdown("<h1 class='main-title'>XGBoost Classifier Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Predict the cluster based on HR, EDA, TEMP, and BVP values</p>", unsafe_allow_html=True)
st.markdown("<p class='instruction-text'>Enter feature values below and click **Predict** to see the result:</p>", unsafe_allow_html=True)


# Input fields for each feature
feature_1 = st.number_input("Enter EDA (Electrodermal Activity)", min_value=-50.0, max_value=100.0, value=0.42)
feature_2 = st.number_input("Enter TEMP (Temperature)", min_value=0.0, max_value=150.0, value=33.3)
feature_3 = st.number_input("Enter BVP (Blood Volume Pulse)", min_value=-60.0, max_value=100.0, value=-18.0)
feature_4 = st.number_input("Enter HR (Heart Rate)", min_value=0.0, max_value=200.0, value=177.0)

# Collect all feature values in a DataFrame for scaling
user_input = pd.DataFrame([[feature_1, feature_2, feature_3, feature_4]], columns=['EDA', 'TEMP', 'BVP','HR'])

# Display input data in the main page
st.markdown("<p class='sub-title'>Selected Input Values:</p>",unsafe_allow_html=True)
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
