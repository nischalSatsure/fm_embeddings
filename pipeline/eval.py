from omegaconf import DictConfig, OmegaConf
import hydra
from src.inference.rf import AEFPredictor
import joblib
import logging

logging.basicConfig(level=logging.INFO)

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def evaluate_forest_cover(cfg: DictConfig):

    model = joblib.load(cfg.model_path)

    predictor = AEFPredictor(model, cfg.year)
    results = predictor.evaluate_predictions(
        eval_raster=cfg.eval_raster,
        inference_region=cfg.inference_region,
        max_workers=cfg.max_workers,
        no_grids=cfg.no_grids
    )
    logging.info(f"Evaluation Results: {results}")


if __name__ == "__main__":
    evaluate_forest_cover()