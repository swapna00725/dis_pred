import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from imblearn.over_sampling import RandomOverSampler

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Data ingestion started")

            # ✅ Read dataset
            df = pd.read_csv("D:/projects_practice/disease_pred/notebook/data/improved_disease_dataset.csv")

            # ✅ Ensure artifacts folder exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # ✅ Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # ✅ Balance dataset
            X = df.drop('disease', axis=1)
            y = df['disease']
            ros = RandomOverSampler(random_state=42)
            X_samp, y_samp = ros.fit_resample(X, y)
            df_balanced = pd.concat([X_samp, y_samp], axis=1)

            # ✅ Train-test split
            train_set, test_set = train_test_split(df_balanced, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Train, test, and raw data saved successfully")
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

        except Exception as e:
            logging.error("Error occurred in data ingestion step")
            raise CustomException(e, sys)


