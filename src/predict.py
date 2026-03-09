import joblib
import numpy as np

# Carica l'INTERA pipeline
model = joblib.load('model/customer_voice_model.pkl')

def analyze_review(text):
    prediction = model.predict([text])[0] #sklearn lavora ad array quindi trasformiamo il testo (1d) in array (2d) e prendiamo [0] ovvero il testo
    
    sentiment = "POSITIVO" if prediction == 1 else "NEGATIVO"
    decision_score = model.decision_function([text])[0]
    
    print(f"\nRecensione: {text}")
    print(f"Sentiment Predetto: {sentiment} (Confidence Score: {decision_score:.2f})")

# Test
analyze_review("The package arrived broken and the food tastes like plastic. Never again!")
analyze_review("Absolutely delicious! My kids loved these snacks, will buy more.")