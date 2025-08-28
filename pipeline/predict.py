from omegaconf import DictConfig, OmegaConf
import hydra

import gc
import numpy as np
from mapminer import miners
# from concurrent.futures import ThreadPoolExecutor, as_completed
from src.predict.rf_batch_predict import AEFPredictor
import joblib
import logging

logging.basicConfig(level=logging.INFO)

@hydra.main(version_base="1.3", config_path="../configs", config_name="predict")
def predict_forest_cover(cfg: DictConfig):

    model = joblib.load(cfg.model_path)
    
    predictor = AEFPredictor(model, cfg.year)
    predictions = predictor.run_predictions(
        gdf_inference_path=cfg.gdf_inference_path,
        batch_size=cfg.batch_size,
        max_workers=cfg.max_workers,
        no_grids=cfg.no_grids
    )

    results = predictor.evaluate_predictions(predictions, cfg.datapath)

    logging.info(f"Evaluation Results: {results}")

if __name__ == "__main__":
    predict_forest_cover()