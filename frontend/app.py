import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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
print(user_input)

# Scale the input data
# array_2d = user_input.to_numpy()
# st.write('2D NumPy Array:')
# st.write(array_2d)
# user_input_reshape = user_input.reshape(-1,1)
user_input_reshape = user_input.to_numpy().reshape(-1, 1)

user_input_scaled = scaler.fit_transform(user_input_reshape)
print(user_input_scaled)
single_row_array = user_input_scaled.reshape(1, -1)
print(single_row_array)


# Button to make a prediction
if st.button("Predict"):
    # Predict the cluster
    prediction = model.predict(single_row_array)
    
    # Display the result with a success message
    st.markdown("<h2 style='color: #4CAF50;'>Prediction Result:</h2>", unsafe_allow_html=True)
    st.write(f"#### The predicted cluster is: **{int(prediction[0])}**")
    st.success("Prediction successful! 🎉")

# Footer with contact info
st.markdown("<hr>", unsafe_allow_html=True)
st.write("Made with ❤️ by [Your Name](https://github.com/YourProfile)")

# Run the app using: streamlit run <filename>.py
