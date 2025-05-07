import mlflow
import mlflow.sklearn
import pandas as pd


model_path = "models/random_forest_model.pkl"  # Modify as per your model location


loaded_model = mlflow.sklearn.load_model("models:/TitanicBestModel/Production")

sample_data = pd.DataFrame({
    'Pclass': [1],
    'Sex': [0], 
    'Age': [22],
    'SibSp': [1],
    'Parch': [0],
    'Fare': [7.25],
    'Embarked': [0] 
})
 
try:
    prediction = loaded_model.predict(sample_data)
    print("🎯 Prediction (Survived=1 / Not Survived=0):", prediction[0])
except Exception as e:
    print("❌ Prediction failed:", str(e))
