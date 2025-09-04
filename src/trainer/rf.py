import dask.dataframe as dd
import pandas as pd
from dask_ml.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
import joblib
import logging

logger = logging.getLogger(__name__)

class RF_Trainer:
    def __init__(self, config):
        """
        config.data should be a dict of category_name -> list of parquet paths
        e.g.
        data:
          woodland: [woodland_1.parquet, woodland_2.parquet]
          open_forest: [open_forest_1.parquet]
          closed_forest: [closed_forest_1.parquet, closed_forest_2.parquet]

        config.split is a dict with keys like test_size, random_state
        config.model is optional dict with n_estimators, n_jobs
        config.paths.report & config.paths.model are output file paths
        """
        self.config = config
        self.model = RandomForestClassifier(
            n_estimators=config.model["n_estimators"],
            n_jobs=config.model["n_jobs"],
        )
        self.background = self.config['background_class']

    def read_data(self):
        """Read all parquet files for each category and assign class labels."""
        dfs = []
        # Assign numeric labels automatically
        class_map = {}

        for label, (class_name, paths) in enumerate(self.config.data.items()):
            df = dd.read_parquet(paths).assign(class_=label)
            class_map[class_name] = label
            dfs.append(df)

        self.data = dd.concat(dfs)
        logger.info(f"Loaded data for classes: {class_map}")

    def preprocess(self, background=5):
        """Drop rows with NaNs, add 50 zero rows with class_=background, fill remaining NaNs with 0."""
        feature_cols = [c for c in self.data.columns if c != "class_"]

        # Drop rows with any NaNs in features
        self.data = self.data.dropna()

        # Add 50 zero rows with class 5
        zero_rows = pd.DataFrame(
            0, index=range(50), columns=feature_cols
        )
        zero_rows["class_"] = background
        zero_rows_dd = dd.from_pandas(zero_rows, npartitions=1)

        # Combine
        self.data = dd.concat([self.data, zero_rows_dd])

    def split_data(self):
        """Split into train/test using dask-ml."""
        X = self.data.drop(columns=["class_"])
        y = self.data["class_"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.split["test_size"],
            shuffle=self.config.split["shuffle"],
            random_state=self.config.split["random_state"],
        )

        self.train_data = (X_train, y_train)
        self.test_data = (X_test, y_test)
        logger.info("Data split into train and test sets.")

    def validate_data(self):
        """Check for NaNs (computes small counts)."""
        sample = self.test_data[0].sample(frac=0.001).compute()
        assert sample.isna().sum().sum() == 0, "Testing features contain NaN values"

        sample = self.test_data[1].sample(frac=0.001).compute()
        assert sample.isna().sum().sum() == 0, "Testing labels contain NaN values"

        sample = self.train_data[0].sample(frac=0.001).compute()
        assert sample.isna().sum().sum() == 0, "Training features contain NaN values"

        sample = self.train_data[1].sample(frac=0.001).compute()
        assert sample.isna().sum().sum() == 0, "Training labels contain NaN values"

        logger.info("Data validation passed: No NaN values found in training and testing datasets.")
        logger.info(f"{len(self.train_data[0])} rows sent for training.")

    def train(self):
        X_train, y_train = self.train_data
        self.model.fit(X_train.compute(), y_train.compute())

    def metrics(self):
        """Compute accuracy and classification report."""
        X_test, y_test = self.test_data
        y_pred = self.model.predict(X_test)

        # Bring metrics into memory
        y_test_np = y_test.compute()
        y_pred_np = y_pred

        acc = accuracy_score(y_test_np, y_pred_np)
        report = classification_report(y_test_np, y_pred_np)

        # Save the classification report to a file
        Path(self.config.paths.report).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.paths.report, "w") as f:
            f.write(report)

        logger.info(f"Model accuracy: {acc:.4f}")
        return {"accuracy": acc, "classification_report": report}

    def save_model(self):
        """Save the trained model."""
        Path(self.config.paths.model).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.config.paths.model)
        logger.info(f"Model saved to {self.config.paths.model}")
