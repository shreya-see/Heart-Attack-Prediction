import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter

# Load the dataset
data = pd.read_csv("processed_heart_data.csv")

# Define top features
top_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'thal']

# Prepare the data
X = data[top_features]
y = data.iloc[:,-1]

num_before = dict(Counter(y))
sampling_strategy = {
    0: 500,  
    1: 500,  
    2: 500,  
}

over = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
under = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

X_smote, y_smote = pipeline.fit_resample(X, y)
num_after = dict(Counter(y_smote))
new_data = pd.concat([pd.DataFrame(X_smote), pd.DataFrame(y_smote)], axis=1)
new_data.columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'thal','num']
x2 = new_data[top_features]
y2= new_data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(x2, y2, test_size=0.2, 
                                                    random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)  # Adjust parameters as needed
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rfc_accuracy = accuracy_score(y_test, y_pred)

# Streamlit form interface
st.title("🫀 Heart Attack Risk Prediction")
st.markdown("""
    **Welcome!** This tool uses a RandomForestClassifier to predict the risk level of heart attack based on several medical features.
    Please fill in the following details to get your result.
""")

# Calculate the dataset size after SMOTE
num_after_smote = len(y_smote)

st.sidebar.title("Model Details")
st.sidebar.markdown("""
    - **Model**: RandomForestClassifier
    - **Dataset Size (before SMOTE)**: {}
    - **Dataset Size (after SMOTE)**: {}
    - **Model Accuracy**: {:.2f}%
    - **Top Features**: Age, Resting BP, Cholesterol, Max HR, ST Depression, Major Vessels, Thalassemia
""".format(len(y), num_after_smote, rfc_accuracy * 100))

# Add custom color for the background
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.form("prediction_form"):
    st.subheader("Enter Patient Information")
    age = st.number_input('Age', min_value=0, help="Enter patient's age")
    trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=0, help="Patient's resting blood pressure")
    chol = st.number_input('Cholesterol Level (mg/dL)', min_value=0, help="Serum cholesterol in mg/dL")
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0, help="Maximum heart rate achieved")
    oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, help="ST depression induced by exercise relative to rest")
    ca = st.number_input('Number of Major Vessels (0-4)', min_value=0, max_value=4, help="Number of major vessels (0-4) colored by fluoroscopy")
    thal = st.selectbox('Thalassemia Type', [3, 6, 7], format_func=lambda x: {3: 'Normal', 6: 'Fixed Defect', 7: 'Reversible Defect'}.get(x))
    
    # Submit button
    submit_button = st.form_submit_button(label='Predict 🧑‍⚕️')

    if submit_button:
        # Check if any input is missing
        if any(v is None or v == '' for v in [age, trestbps, chol, thalach, oldpeak, ca, thal]):
            st.warning("⚠️ Please fill in all fields to get a prediction.")
        else:
            # Prepare input
            user_input = np.array([[age, trestbps, chol, thalach, oldpeak, ca, thal]])

            # Predict the heart attack level
            prediction = model.predict(user_input)[0]
            
            # Add visual feedback
            st.subheader("Prediction Result")
            
            # Define risk levels with color coding
            if prediction == 0:
                st.success("🟢 No risk of heart attack. Predicted risk level: {}".format(prediction))
            elif 1 <= prediction <= 2:
                st.warning("🟡 Low risk of heart attack. Predicted risk level: {}".format(prediction))
            elif 3 <= prediction <= 4:
                st.error("🔴 High risk of heart attack. Predicted risk level: {}".format(prediction))
            else:
                st.error("🔴 Uncertain risk level. Predicted risk level: {}".format(prediction))
            
            st.markdown("""
            *Please note*: This prediction is based on the input features and does not replace professional medical advice.
            """)
