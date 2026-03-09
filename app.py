import streamlit as st
import joblib

# Configurazione pagina
st.set_page_config(page_title="Customer Voice Insights", page_icon="📊")

st.title("🚀 Customer Voice Insights Engine")
st.markdown("""
Analizza il sentiment delle recensioni dei tuoi clienti in tempo reale. 
Questo strumento identifica se una recensione è un segnale di **Fedeltà** o un rischio di **Abbandono (Churn)**.
""")

# Caricamento modello
@st.cache_resource # Evita di ricaricare il modello a ogni click
def load_model():
    return joblib.load('model/customer_voice_model.pkl')

pipeline = load_model()

# Area di input
review_text = st.text_area("Incolla qui la recensione del cliente:", 
                           placeholder="Esempio: The product arrived late and the quality is poor...")

if st.button("Analizza Sentiment"):
    if review_text.strip():
        # Predizione
        prediction = pipeline.predict([review_text])[0]
        score = pipeline.decision_function([review_text])[0]
        
        # Visualizzazione Risultati
        st.divider()
        if prediction == 1:
            st.success(f"### Sentiment: POSITIVO ✅")
            st.balloons()
        else:
            st.error(f"### Sentiment: NEGATIVO 🚩")
            
        st.write(f"**Confidence Score:** {score:.2f}")
        st.info("Un punteggio positivo indica soddisfazione, uno negativo indica rischio di abbandono.")
    else:
        st.warning("Per favore, inserisci del testo per l'analisi.")

# Sidebar con info tecniche
st.sidebar.title("Model Info")
st.sidebar.info("""
- **Model:** LinearSVC
- **Vectorization:** TF-IDF (1,2-grams)
- **Dataset:** Amazon Fine Food Reviews
""")