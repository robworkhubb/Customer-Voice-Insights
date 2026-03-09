import streamlit as st
import joblib

# Configurazione pagina
st.set_page_config(page_title="Customer Voice Insights", page_icon="📊")

st.title("🚀 Customer Voice Insights Engine")
st.markdown("""
Analyze the sentiment of your customer reviews in real time.
This tool identifies whether a review is a signal of **Loyalty** or a risk of **Churn**.
""")

# Caricamento modello
@st.cache_resource # Evita di ricaricare il modello a ogni click
def load_model():
    return joblib.load('model/customer_voice_model.pkl')

pipeline = load_model()

# Area di input
review_text = st.text_area("Paste customer review here:", 
                           placeholder="Esempio: The product arrived late and the quality is poor...")

if st.button("Analizza Sentiment"):
    if review_text.strip():
        # Predizione
        prediction = pipeline.predict([review_text])[0]
        score = pipeline.decision_function([review_text])[0]
        
        # Visualizzazione Risultati
        st.divider()
        if prediction == 1:
            st.success(f"### Sentiment: POSITIVE ✅")
            st.balloons()
        else:
            st.error(f"### Sentiment: NEGATIVE 🚩")
            
        st.write(f"**Confidence Score:** {score:.2f}")
        st.info("A positive score indicates satisfaction, a negative one indicates risk of abandonment.")
    else:
        st.warning("Please enter some text for analysis.")

# Sidebar con info tecniche
st.sidebar.title("Model Info")
st.sidebar.info("""
- **Model:** LinearSVC
- **Vectorization:** TF-IDF (1,2-grams)
- **Dataset:** Amazon Fine Food Reviews
""")