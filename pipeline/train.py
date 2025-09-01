from src.trainer.rf import RF_Trainer
import hydra
from omegaconf import DictConfig
import logging


logger = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def train_model(cfg: DictConfig) -> None:

    logger.info("Initializing RF_ModelHandler with config.")
    model_handler = RF_Trainer(cfg)

    logger.info("Reading data.")
    model_handler.read_data()

    logger.info("Splitting data.")
    model_handler.split_data()

    logger.info("Training model.")
    model_handler.train()

    logger.info("Calculating metrics.")
    metrics = model_handler.metrics()
    logger.info(f"Model metrics: {metrics}")

    logger.info("Saving model.")
    model_handler.save_model()

    logger.info("Training pipeline completed successfully.")

@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def exp_train_model(cfg: DictConfig) -> None:
    logger.info("Initializing RF_ModelHandler with config.")
    model_handler = RF_Trainer(cfg)

    logger.info("Reading data.")
    model_handler.read_data()

    logger.info("Preprocessing data.")
    model_handler.preprocess()

    logger.info("Splitting data.")
    model_handler.split_data()

    logger.info("Validate data")
    model_handler.validate_data()

    logger.info("Training model.")
    model_handler.train()

    logger.info("Calculating metrics.")
    metrics = model_handler.metrics()
    logger.info(f"Model metrics: {metrics}")

    logger.info("Saving model.")
    model_handler.save_model()

    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    exp_train_model()