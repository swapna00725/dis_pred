import sys
import os
import pandas as pd
import numpy as np
from statistics import mode
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        try:
            # Load models & preprocessor once during initialization
            model_path = os.path.join("artifacts", "disease_predictor.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            loaded_data = load_object(file_path=model_path)
            self.preprocessor = load_object(file_path=preprocessor_path)

            # Store models and symptom index
            self.rf_model = loaded_data.get("rf_model")
            self.knn_model = loaded_data.get("knn_model")  # fixed key
            self.dt_model = loaded_data.get("dt_model")
            self.symptom_index = loaded_data.get("symptom_index", {})
            
        except Exception as e:
            raise CustomException(e, sys)

    def predict_disease(self, input_symptoms):
        """For predicting from comma-separated symptoms."""
        try:
            input_symptoms = input_symptoms.split(",")
            input_symptoms = [s.strip() for s in input_symptoms]

            # Filter valid symptoms
            valid_symptoms = [s for s in input_symptoms if s in self.symptom_index]

            if not valid_symptoms:
                return None, None, None, "No valid symptoms found in dataset."

            # Create input vector
            input_data = [0] * len(self.symptom_index)
            for symptom in valid_symptoms:
                input_data[self.symptom_index[symptom]] = 1
            input_data = np.array(input_data).reshape(1, -1)

            # Predictions from all models
            rf_pred = self.rf_model.predict(input_data)[0]
            knn_pred = self.knn_model.predict(input_data)[0]
            dt_pred = self.dt_model.predict(input_data)[0]

            final_pred = mode([str(rf_pred), str(knn_pred), str(dt_pred)])

            return rf_pred, knn_pred, dt_pred, final_pred

        except Exception as e:
            raise CustomException(e, sys)