from sklearn.metrics import classification_report, confusion_matrix
from model import pipeline
from model import X_test, y_test

y_pred = pipeline.predict(X_test)

print("--- Report di Classificazione ---")
print(classification_report(y_test, y_pred))