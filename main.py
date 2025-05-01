import hydra
from omegaconf import DictConfig
from src.logger import ExecutorLogger
from src.training.model_evaluate import evaluate
from src.training.data_training import extract_target, trainer
from src.training.process_data import read_and_process_data
from sklearn.ensemble import RandomForestClassifier
from src.training.data_training import extract_target, trainer
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

import logging
import hydra
from omegaconf import DictConfig



@hydra.main(config_path="config", config_name="config_random_forest", version_base=None)
def main(cfg: DictConfig):
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Ensure logs appear in the console
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.info("Pipeline started.")

    # Extract the target variable
    X_train, y_train, X_test, y_test = extract_target("titanic_data", "Survived", logger)

    # Train the model
    model = trainer(X_train, y_train, logger)

    # Evaluate the model and log the results
    y_pred = model.predict(X_test)
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    logger.info(f"Classification Report: \n{classification_report(y_test, y_pred)}")
