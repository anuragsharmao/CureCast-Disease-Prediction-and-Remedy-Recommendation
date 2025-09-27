from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import json
import re
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load datasets
DATA_FILE = 'disease_data.csv'
REMEDIES_FILE = 'remedies.json'

# Sample stopwords (minimal list)
STOPWORDS = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs', 'have', 'has', 'had', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'a', 'an', 'the', 'and', 'but', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}

# Global variables for model and symptoms
model = None
symptom_list = []
label_encoder = None
remedies = {}
severity_colors = {'Common Cold': 'green', 'Flu': 'orange', 'Migraine': 'orange'}  # Simple color mapping

def train_model():
    global model, symptom_list, label_encoder
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found. Create it with sample data.")
    
    df = pd.read_csv(DATA_FILE)
    symptom_list = [col for col in df.columns if col != 'disease']
    
    X = df[symptom_list].values
    y = df['disease'].values
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y_encoded)
    print("Model trained successfully.")

def load_remedies():
    global remedies
    if not os.path.exists(REMEDIES_FILE):
        raise FileNotFoundError(f"{REMEDIES_FILE} not found. Create it with sample data.")
    
    with open(REMEDIES_FILE, 'r') as f:
        remedies = json.load(f)

# Preprocess text and extract matched symptoms
def preprocess_and_extract(symptoms_text):
    # Lowercase, remove punctuation, strip, split
    text = re.sub(r'[^\w\s]', '', symptoms_text.lower().strip())
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    
    # Binary vector: 1 if symptom in words
    vector = np.zeros(len(symptom_list))
    matched = []
    for i, symptom in enumerate(symptom_list):
        if symptom.lower() in words:
            vector[i] = 1
            matched.append(symptom)
    
    return vector, matched

# Train on startup
train_model()
load_remedies()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    return jsonify({'symptoms': symptom_list})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms_text = data.get('symptoms', '').strip()
    if not symptoms_text:
        return jsonify({'error': 'No symptoms provided'}), 400
    
    vector, matched = preprocess_and_extract(symptoms_text)
    prediction_encoded = model.predict(vector.reshape(1, -1))[0]
    disease = label_encoder.inverse_transform([prediction_encoded])[0]
    
    return jsonify({
        'disease': disease,
        'matched_symptoms': matched,
        'confidence': float(model.predict_proba(vector.reshape(1, -1)).max())  # Optional explainability
    })

@app.route('/remedies', methods=['POST'])
def get_remedies():
    data = request.json
    disease = data.get('disease', '').strip()
    if disease not in remedies:
        return jsonify({'error': 'Unknown disease'}), 404
    
    remedy = remedies[disease]
    color = severity_colors.get(disease, 'gray')
    
    return jsonify({
        'disease': disease,
        'remedy': remedy,
        'color': color
    })

if __name__ == '__main__':
    app.run(debug=True)
