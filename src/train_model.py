from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


def train_model(train_df):
    mlflow.set_tracking_uri("https://dagshub.com/naaadiaaa/MLOps_labs.mlflow")

    X = train_df.drop(columns=["Survived"])
    y = train_df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    client = MlflowClient()

    best_accuracy = -1
    best_model = None

    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_param("model_type", name)
            if name == "random_forest":
                mlflow.log_param("n_estimators", 100)
            elif name == "logistic_regression":
                mlflow.log_param("max_iter", 1000)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", acc)

            mlflow.sklearn.log_model(model, "model", registered_model_name="TitanicBestModel")

            if acc > best_accuracy:
                best_accuracy = acc
                best_model = model
                best_run_id = run.info.run_id

    
    latest_versions = client.get_registered_model("TitanicBestModel").latest_versions
    best_version = None
    for version in latest_versions:
        if version.run_id == best_run_id:
            best_version = version
            break

    if best_version:
        client.transition_model_version_stage(
            name="TitanicBestModel",
            version=best_version.version,
            stage="Production"
        )
        print(f"✅ Model version {best_version.version} promoted to Production.")
    else:
        print("❌ Could not identify best model version to promote.")

    return best_model
