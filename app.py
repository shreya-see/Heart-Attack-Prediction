from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Initialize Flask app
app = Flask(__name__)

# Load the dataset
data = pd.read_csv("processed_heart_data.csv")

# Define top features based on Boruta feature selection
top_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'thal']

# Prepare the data
X = data[top_features]
y = data['num']

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Define home route that displays the form
@app.route('/')
def home():
    return render_template('index.html')

# Define route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the form
        age = float(request.form['age'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        thalach = float(request.form['thalach'])
        oldpeak = float(request.form['oldpeak'])
        ca = float(request.form['ca'])
        thal = float(request.form['thal'])
        
        # Prepare the input data for prediction
        user_input = np.array([[age, trestbps, chol, thalach, oldpeak, ca, thal]])
        
        # Predict the heart attack level
        prediction = model.predict(user_input)
        
        # Return result to the user
        return render_template('index.html', prediction=f"Predicted heart attack level (num): {prediction[0]}")
    
    except Exception as e:
        return render_template('index.html', prediction=f"An error occurred: {str(e)}")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
