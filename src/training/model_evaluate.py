import os
import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report

MODEL_PATH = "models"
REPORT_PATH = "reports"

def evaluate(X_test, y_test, model_path):
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    y_true = y_test

    print("=== Sklearn Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("\n=== Skorecard Classification Summary ===")
    print(classification_report(y_true, y_pred))  
