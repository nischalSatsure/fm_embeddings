
import hydra
from omegaconf import DictConfig, OmegaConf
from src.dataset.aef import AEFDataHandler
import pandas as pd
import logging



@hydra.main(version_base="1.3", config_path="../configs/data", config_name="default")
def test_main(cfg: DictConfig):

    data_handler = AEFDataHandler()
    data_handler.create_dataset_from_polygons(**cfg)
    logging.info(f"Dataset saved to {cfg.output_path}")

    assert pd.read_parquet(cfg.output_path).shape[0] > 0

if __name__ == "__main__":
    test_main()