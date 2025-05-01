from src.training.process_data import read_and_process_data
import logging

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Run preprocessing
read_and_process_data(
    file_name="Titanic",
    id_col="PassengerId",
    target_col="Survived",
    logger=logger
)
