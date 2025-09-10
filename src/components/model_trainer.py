import os
import sys

from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from statistics import multimode

from src.utils import save_object

import pandas as pd
import numpy as np

@dataclass
class ModelTrainerConfig:
    model_file_path=os.path.join("artifacts","disease_predictor.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_config=ModelTrainerConfig()
    def initiate_model_trainer(self,train_arr,test_arr,symptom_index):
        try:
            logging.info("model training has started")
                        
            Xtr, ytr = train_arr[:, :-1], train_arr[:, -1]
            Xte, yte = test_arr[:, :-1], test_arr[:, -1]

            dt_model = DecisionTreeClassifier().fit(Xtr, ytr)
            rf_model = RandomForestClassifier().fit(Xtr, ytr)
            knn_model = KNeighborsClassifier().fit(Xtr, ytr)

            dt_preds = dt_model.predict(Xte)
            rf_preds = rf_model.predict(Xte)
            knn_preds = knn_model.predict(Xte)       

            final_preds = []
            for i, j, k in zip(dt_preds, rf_preds, knn_preds):
                vote = multimode([i, j, k])[0]  # pick first in case of tie
                final_preds.append(vote)

            pkl_data = {
                 "dt_model": dt_model,
                 "rf_model": rf_model,
                 "knn_model": knn_model,
                 "symptom_index": symptom_index
                 }
                    
            save_object(file_path=self.model_config.model_file_path,obj=pkl_data)

            acc_score=accuracy_score(yte, final_preds)

            print(f"Combined Model Accuracy: {acc_score * 100:.2f}%")

            
            return acc_score
        
        except Exception as e:
            raise CustomException(e,sys)
            logging.error("failed in training the model")



