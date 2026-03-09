# 🚀 Customer-Voice-AI: Insights Engine

## 📌 Overview
Questo progetto nasce per risolvere un problema aziendale critico: **capire il "perché" dietro l'insoddisfazione dei clienti.** Utilizzando il dataset *Amazon Fine Food Reviews*, ho costruito una pipeline di Machine Learning capace di analizzare migliaia di recensioni e identificare automaticamente i pattern linguistici correlati al rischio di abbandono (Churn).

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **ML Framework:** Scikit-Learn (Pipeline, LinearSVC, TfidfVectorizer)
* **Data Ops:** Pandas, NumPy, Joblib
* **Visualization:** Matplotlib, Seaborn

## 📈 Performance & Results
Il modello è ottimizzato per la **Recall sulla classe negativa**, assicurando che le critiche dei clienti vengano intercettate con la massima priorità.

| Metric | Score |
|-------|-------|
| **Accuracy** | 91% |
| **Recall (Negative Class)** | 90% |
| **F1-Score (Positive Class)** | 95% |

### 🔍 Actionable Insights
Dall'analisi dei coefficienti del modello sono emersi i seguenti driver:
* 🚩 **Negativi (Churn Risk):** "undrinkable", "deceptive", "worst", "couldn finish".
* ✅ **Positivi (Loyalty):** "highly recommend", "delicious", "won regret".



## 📂 Project Structure
```text
├── data/               # Dataset (escluso da Git)
├── model/             # Modelli serializzati (.pkl)
├── src/                
│   ├── data.py         # Script di preprocessing e labeling
│   ├── model.py        # Training pipeline e valutazione
│   ├── analytics.py    # Estrazione degli insights aziendali
|   ├── evaluation.py   # Valutazione e metriche
│   └── predict.py      # Script di inferenza per nuove recensioni
├── app.py
├── requirements.txt    # Dipendenze del progetto
└── README.md