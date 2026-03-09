import pandas as pd
from model import pipeline

def get_insights(pipeline, n=15):
    # Estraiamo il vettorizzatore e il classificatore dalla pipeline
    tfidf = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']
    
    # Otteniamo i nomi delle parole
    feature_names = tfidf.get_feature_names_out()
    
    # Otteniamo i coefficienti (pesi) assegnati dal LinearSVC
    coefs = clf.coef_[0]
    
    # Creiamo un DataFrame per visualizzarli meglio
    insights = pd.DataFrame({'word': feature_names, 'weight': coefs})
    
    print("\n--- 🚩 TOP DRIVER DI INSODDISFAZIONE (Sentiment Negativo) ---")
    print(insights.sort_values(by='weight').head(n))
    
    print("\n--- ✅ TOP DRIVER DI SODDISFAZIONE (Sentiment Positivo) ---")
    print(insights.sort_values(by='weight', ascending=False).head(n))

# Esegui la funzione
get_insights(pipeline)