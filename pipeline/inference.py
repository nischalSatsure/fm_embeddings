
from omegaconf import DictConfig
import hydra
from src.inference.rf import AEFPredictor
import joblib
import logging

logging.basicConfig(level=logging.INFO)

@hydra.main(version_base="1.3", config_path="../configs", config_name="inference")
def create_forest_cover(cfg: DictConfig):

    model = joblib.load(cfg.model_path)

    predictor = AEFPredictor(model, cfg.year)

    predictor.save_forest_cover(
        output_path=cfg.output_path,
        gdf_inference_path=cfg.gdf_inference_path,
        max_workers=cfg.max_workers,
        no_grids=cfg.no_grids
    )

    logging.info(f"Forest cover map saved to {cfg.output_path}")

if __name__ == "__main__":  
    create_forest_cover()
