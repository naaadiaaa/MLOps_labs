
import pickle
import mlflow.sklearn
import joblib

def save_model(model, output_path):
     
    joblib.dump(model, output_path)

