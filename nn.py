from flask import Flask, request, jsonify
import os
import PyPDF2
import nltk
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

app = Flask(__name__)

# Dummy dataset for training the model
# You should replace this with a real dataset for better results
train_data = [
    {'cv': 'I enjoy leading teams and working in social environments. I am a team player.', 'leadership_preference': 'leader', 'social_preference': 'gatherings', 'label': 1},  # Extrovert
    {'cv': 'I prefer working independently and in quiet environments. I am a solitary worker.', 'leadership_preference': 'independent', 'social_preference': 'quiet', 'label': 0},  # Introvert
]

# Preprocessing function to extract text and responses
def preprocess_data(data):
    vectorizer = CountVectorizer(stop_words='english')
    cv_texts = [item['cv'] for item in data]
    X_cv = vectorizer.fit_transform(cv_texts)  # Convert CV text to feature vectors
    
    X_aptitude = np.array([[1 if item['leadership_preference'] == 'leader' else 0,
                            1 if item['social_preference'] == 'gatherings' else 0] for item in data])
    
    X = np.hstack([X_cv.toarray(), X_aptitude])  # Combine CV features with aptitude test features
    y = [item['label'] for item in data]
    
    return X, y, vectorizer

# Prepare the model
X, y, vectorizer = preprocess_data(train_data)
model = LogisticRegression()
model.fit(X, y)

@app.route('/predict_personality', methods=['POST'])
def predict_personality():
    if 'cv' not in request.files:
        return jsonify({'error': 'No CV uploaded'}), 400
    
    # Read the CV file (either PDF or TXT)
    cv_file = request.files['cv']
    
    # Extract text from the CV
    if cv_file.filename.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(cv_file)
        cv_text = ""
        for page in pdf_reader.pages:
            cv_text += page.extract_text()
    else:
        cv_text = cv_file.read().decode("utf-8")
    
    # Extract features from the CV text
    cv_features = vectorizer.transform([cv_text]).toarray()
    
    # Process aptitude test answers
    leadership_preference = request.form.get('leadership_preference')
    social_preference = request.form.get('social_preference')
    
    aptitude_features = np.array([[1 if leadership_preference == 'leader' else 0,
                                   1 if social_preference == 'gatherings' else 0]])
    
    # Combine the features from the CV and the aptitude test
    X_test = np.hstack([cv_features, aptitude_features])
    
    # Predict personality (0 = Introvert, 1 = Extrovert)
    prediction = model.predict(X_test)[0]
    personality = 'Extrovert' if prediction == 1 else 'Introvert'
    
    return jsonify({'prediction': personality})

if __name__ == '__main__':
    app.run(debug=True)
