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

from src.utils import save_object,evaluate_models

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
                        # Split data
            Xtr, ytr = train_arr[:, :-1], train_arr[:, -1]
            Xte, yte = test_arr[:, :-1], test_arr[:, -1]

            models={"Decision Tree":DecisionTreeClassifier(),
                    "Random Forest": RandomForestClassifier(),
                    "Kneighbors":KNeighborsClassifier()}
            
            params= { "Decision Tree": {
                     'criterion':['gini','entropy'],
                     'max_depth': [None, 5, 10, 15],
                     'min_samples_split': [2, 5, 10]},
                
                    "Random Forest":{
                    'criterion': ['gini', 'entropy'],
                    'n_estimators': [10, 50, 100],
                    'max_depth': [None, 5, 10],                        
                    },
                
                    "Kneighbors" :{
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['minkowski', 'manhattan']
                    }
                
                   }
            model_report : dict = evaluate_models(Xtr,ytr,Xte,yte,models,params)
            print(model_report)

            dt_model = DecisionTreeClassifier(**model_report["Decision Tree"]["best_params"]).fit(Xtr, ytr)
            rf_model = RandomForestClassifier(**model_report["Random Forest"]["best_params"]).fit(Xtr, ytr)
            knn_model = KNeighborsClassifier(**model_report["KNeighbors"]["best_params"]).fit(Xtr, ytr)

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
                 "knn_mode": knn_model,
                 
                 "symptom_index": symptom_index
                 }
                    
            save_object(self.model_config.model_file_path,pkl_data)

            acc_score=accuracy_score(yte, final_preds)

            print(f"Combined Model Accuracy: {acc_score * 100:.2f}%")

            
            return acc_score
        
        except Exception as e:
            raise CustomException(e,sys)
            logging.error("failed in training the model")



