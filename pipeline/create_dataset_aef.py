
from anyio import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from src.data.aef_fetch import AEFDataHandler
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@hydra.main(version_base="1.3", config_path="../configs/data", config_name="default")
def main(cfg: DictConfig):

    max_workers = cfg.max_workers
    filepath = cfg.aef.filepath
    year = cfg.aef.year
    output_path = cfg.aef.output_path
    clip = cfg.aef.clip
    cls = cfg.aef.cls

    data_handler = AEFDataHandler()
    if Path(output_path).suffix == '.parquet':
        data_handler.create_dataset_from_polygons_parquet(
            max_workers=max_workers,
            filepath=filepath,
            year=year,
            output_path=output_path,
            clip=clip,
            cls=cls
        )
    else:
        data_handler.create_dataset_from_polygons_csv(
            max_workers=max_workers,
            filepath=filepath,
            year=year,
            output_path=output_path,
            clip=clip,
            cls=cls
        )

    logging.info(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    main()
