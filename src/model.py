from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from data import preprocessing
import joblib

X, y = preprocessing('data/Reviews.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        stop_words='english',
        lowercase=True, # Converte tutto in minuscolo
        strip_accents='unicode', # Rimuove accenti (es: è -> e)
        token_pattern=r'\b[a-zA-Z]{3,}\b' # Prende solo parole di almeno 3 lettere, ignorando numeri e punteggiatura
    )),
    ('clf', LinearSVC(C=1.0, class_weight='balanced', random_state=42))
])

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, 'model/customer_voice_model.pkl')

