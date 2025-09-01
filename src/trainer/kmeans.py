from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from pathlib import Path
import pandas as pd
import joblib
import logging


logger = logging.getLogger(__name__)

class Kmeans_Trainer:
    def __init__(self, config, k):
        self.config = config
        self.model = KMeans(n_clusters=k)

    def read_data(self):
        self.forest = pd.read_parquet(self.config.data.forest)
        self.forest["class"] = 1 

        self.non_forest = pd.read_parquet(self.config.data.non_forest)
        self.non_forest["class"] = 0

        self.data = pd.concat([self.forest, self.non_forest])
        del self.forest, self.non_forest

    def preprocess(self):
        nan_rows = self.data[self.data.isna().all(axis=1)]
        limited_nan_rows = nan_rows.head(200).fillna(0)

        self.data = pd.concat([
            self.data[~self.data.isna().all(axis=1)],
            limited_nan_rows], ignore_index=True)

        # Fill any other NaNs left in features
        self.data = self.data.fillna(0)
        del limited_nan_rows, nan_rows

    def split_data(self):
        X = self.data.drop(columns=['class'], axis=1)
        y = self.data['class']

        del self.data

        X_train, X_test, y_train, y_test = train_test_split(X, y, **self.config.split)
        
        del X, y

        self.train_data = (X_train, y_train)
        self.test_data = (X_test, y_test)
    
    def validate_data(self):
        assert self.train_data[0].isna().sum().sum() == 0, "Training features contain NaN values"
        assert self.train_data[1].isna().sum().sum() == 0, "Training labels contain NaN values"
        assert self.test_data[0].isna().sum().sum() == 0, "Testing features contain NaN values"
        assert self.test_data[1].isna().sum().sum() == 0, "Testing labels contain NaN values"

        logger.info("Data validation passed: No NaN values found in training and testing datasets.")
        logger.info(f"{len(self.train_data[0])} rows sent for training.")

    def train(self):
        self.model.fit(*self.train_data)

    def metrics(self):
        X_test, y_test = self.test_data
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Save the classification report to a file
        Path(self.config.report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.report_path, "w") as f:
            f.write(report)

        return {"accuracy": acc, "classification_report": report}

    def save_model(self):
        Path(self.config.model_save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.config.model_save_path)
