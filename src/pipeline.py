import argparse
import yaml
import os

from data_loader import load_data
from preprocessing import wrangle
from save_model import save_model
from train_model import train_model

def main(config_path):

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    print("Model output will be saved to:", cfg["paths"]["model_output"])

    
    train_df = load_data(cfg["paths"]["raw_file"])
    train_df_preprocessed = wrangle(train_df)


    predictor = train_model(train_df_preprocessed)

    
    model_dir = os.path.dirname(cfg["paths"]["model_output"])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    save_model(predictor, cfg["paths"]["model_output"])


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    
    
    main(args.config)
