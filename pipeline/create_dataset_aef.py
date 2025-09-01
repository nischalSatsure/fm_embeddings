
from anyio import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from src.data.aef_fetch import AEFDataHandler
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@hydra.main(version_base="1.3", config_path="../configs", config_name="dataset")
def main(cfg: DictConfig):

    max_workers = cfg.max_workers
    filepath = cfg.dataset.filepath
    year = cfg.dataset.year
    output_path = cfg.dataset.output_path
    clip = cfg.dataset.clip
    cls = cfg.dataset.cls

    # data_handler = AEFDataHandler()
    # if Path(output_path).suffix == '.parquet':
    #     data_handler.create_dataset_from_polygons_parquet(
    #         max_workers=max_workers,
    #         filepath=filepath,
    #         year=year,
    #         output_path=output_path,
    #         clip=clip,
    #         cls=cls
    #     )
    # else:
    #     data_handler.create_dataset_from_polygons_csv(
    #         max_workers=max_workers,
    #         filepath=filepath,
    #         year=year,
    #         output_path=output_path,
    #         clip=clip,
    #         cls=cls
    #     )

    # logging.info(f"Dataset saved to {output_path}")
    print(cfg)
if __name__ == "__main__":
    main()
