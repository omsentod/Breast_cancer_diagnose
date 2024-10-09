from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler  # Menggunakan MinMaxScaler

app = Flask(__name__)

# Load dataset
file_path = 'UTS/kankerpentil-preprocessing-not.id.csv'
data = pd.read_csv(file_path)

# Assume target column is 'diagnosis' and features start from the 3rd column
X = data.iloc[:, 2:].values  # Features
y = data['diagnosis'].values  # Target

# Transform features using Min-Max scaling
scaler = MinMaxScaler()  # Menggunakan MinMaxScaler
X = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train Decision Tree Model
decision_tree = DecisionTreeClassifier(random_state=0, max_depth=3)
decision_tree.fit(X_train, y_train)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diagnosa')
def diagnosa():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form
        input_features = [float(request.form[key].replace(',', '.')) for key in request.form.keys()]
        
        # Normalize input using Min-Max scaling
        input_array = scaler.transform([input_features])  # Menggunakan MinMaxScaler untuk normalisasi input
        pred = decision_tree.predict(input_array)[0]  # Menggunakan decision tree untuk prediksi
        
        # Translate prediction result
        hasil_prediksi = "Malignant" if pred == 1 else "Benign"
        
        # Display result on result page
        return render_template('result.html', prediction=hasil_prediksi)
    except Exception as e:
        return f"An error occurred: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)
