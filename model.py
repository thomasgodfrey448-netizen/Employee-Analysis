import streamlit as st
import pandas as pd
import pickle
import os

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'random_forest_model.pkl')
with open(model_path, 'rb') as f:
    rf_model = pickle.load(f)

# Page Configuration
st.set_page_config(
    page_title="Employee Performance Prediction",
    page_icon="📊",
    layout="wide"
)

# Title and Description
st.title("📊 Employee Performance Prediction System")
st.markdown("""
This application predicts an employee's performance rating based on key workplace and career-related factors.
Adjust the input values below and click **Predict Performance** to generate a prediction.
""")

st.divider()

# Feature Input Section
st.subheader("Employee Information")

col1, col2 = st.columns(2)

env_map = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Very High": 4
}

with col1:
    env_satisfaction = st.selectbox(
        "Environment Satisfaction",
        ["Low", "Medium", "High", "Very High"],
        help="Employee's satisfaction with the work environment."
    )

    salary_hike = st.slider(
        "Last Salary Hike (%)",
        min_value=11,
        max_value=25,
        value=15
    )

    current_role_years = st.slider(
        "Years in Current Role",
        min_value=0,
        max_value=18,
        value=4
    )

with col2:
    company_years = st.slider(
        "Years at Current Company",
        min_value=0,
        max_value=40,
        value=7
    )

    years_since_promotion = st.slider(
        "Years Since Last Promotion",
        min_value=0,
        max_value=15,
        value=2
    )

# Create Input DataFrame
input_data = pd.DataFrame({
    'EmpEnvironmentSatisfaction': [env_map[env_satisfaction]],
    'EmpLastSalaryHikePercent': [salary_hike],
    'ExperienceYearsInCurrentRole': [current_role_years],
    'ExperienceYearsAtThisCompany': [company_years],
    'YearsSinceLastPromotion': [years_since_promotion]
})

st.divider()

# Preview Inputs
with st.expander("View Input Data"):
    st.dataframe(input_data, use_container_width=True)

# Prediction Button
if st.button("Predict Performance", use_container_width=True):
    
    prediction = rf_model.predict(input_data)[0]
    probability = rf_model.predict_proba(input_data).max() * 100

    rating_map = {
        2: "Low Performance",
        3: "Good Performance",
        4: "Excellent Performance"
    }
    rating_label = rating_map.get(prediction, str(prediction))

    st.success(f"Predicted Performance Rating: **{rating_label}**")
    st.metric(
        label="Prediction Confidence",
        value=f"{probability:.2f}%"
    )