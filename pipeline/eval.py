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

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def evaluate_forest_cover(cfg: DictConfig):

    model = joblib.load(cfg.model_path)

    predictor = AEFPredictor(model, cfg.year)
    results = predictor.evaluate_predictions(
        gdf_inference_path=cfg.gdf_inference_path,
        max_workers=cfg.max_workers,
        no_grids=cfg.no_grids
    )

    logging.info(f"Evaluation Results: {results}")


if __name__ == "__main__":
    evaluate_forest_cover()
    # create_forest_cover()