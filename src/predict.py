import mlflow
import pandas as pd
import dagshub
import joblib


"""
def setup_dagshub() -> None:
  
    dagshub.auth.add_app_token(token="dfcea7c9d499ffc20a10f90c1d9d1336c02055b7")
    dagshub.init(
        repo_owner ="naaadiaaa",
        repo_name="MLOps_labs",
        mlflow=True,
    )

setup_dagshub()
logged_model = 'runs:/201eecce97dd4fcc8f5d10e363d5818e/models'

# Load model as a PyFuncModel.
loaded_model = mlflow.sklearn.load_model(logged_model)

"""


#with open ("models/random_forest_model.pkl","rb") as pkl:
   # loaded_model= pickle.load(pkl)


loaded_model = joblib.load("models/random_forest_model.pkl")
 
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
