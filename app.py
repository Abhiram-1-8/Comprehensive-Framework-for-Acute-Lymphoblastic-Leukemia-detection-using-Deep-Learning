from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import cv2
import numpy as np
import pyttsx3
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained .h5 model
model = tf.keras.models.load_model('leukemia_model.h5')

# Simulated user storage (No SQLite)
users = {}

def voice_notification(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        if name in users:
            return "User already exists!"
        users[name] = {'email': email, 'password': password}
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        name = request.form['name']
        password = request.form['password']
        if name in users and users[name]['password'] == password:
            session['user'] = name
            voice_notification("You have successfully logged in")
            return redirect(url_for('demo'))
        return "Invalid credentials!"
    return render_template('login.html')

@app.route('/demo', methods=['GET', 'POST'])
def demo():
    if 'user' not in session:
        return redirect(url_for('login'))
    prediction = None
    precautions = None  # Ensure precautions is always defined
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded!"
        file = request.files['file']
        if file.filename == '':
            return "No selected file!"
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        prediction = predict_leukemia(filepath)
        precautions = get_precautions(prediction)
    return render_template('demo.html', prediction=prediction, precautions=precautions)

def predict_leukemia(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Ensure correct input shape
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Adjust for model input
    prediction = model.predict(img)
    class_labels = ['Non-Cancer', 'Benign', 'Malignant Pre-B', 'Malignant Pro-B', 'Malignant Early Pre-B']
    predicted_class = class_labels[np.argmax(prediction)]
    return predicted_class

def get_precautions(prediction):
    precautions_dict = {
        'Non-Cancer': "Maintain a healthy lifestyle and have regular check-ups.",
        'Benign': "Monitor condition, maintain a balanced diet, and follow up with a doctor.",
        'Malignant Pre-B': "Early-stage treatment advised, consult a specialist, and maintain good nutrition.",
        'Malignant Pro-B': "Undergo chemotherapy/radiotherapy as recommended, ensure high protein intake, and regular medical check-ups.",
        'Malignant Early Pre-B': "Immediate medical attention required, advanced treatment options, and strong immune support diet."
    }
    return precautions_dict.get(prediction, "No specific precautions available.")

@app.route('/precautions')
def precautions():
    if 'user' not in session:
        return redirect(url_for('login'))
    prediction = request.args.get('prediction', 'Non-Cancer')
    precautions = get_precautions(prediction)
    return render_template('precautions.html', precautions=precautions)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
