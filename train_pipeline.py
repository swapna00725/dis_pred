import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.exception import CustomException 
from src.logger import logging


if __name__=="__main__":
    try:
        logging.info("training pipelining started")

        ingestion=DataIngestion()
        tr_data,te_data=ingestion.initiate_data_ingestion()

        transformer=DataTransformation()
        tr_arr,te_arr,symptom_index,_=transformer.initiate_data_transformation(tr_data,te_data)

        trainer=ModelTrainer()
        acc_score=trainer.initiate_model_trainer(tr_arr,te_arr,symptom_index)
        logging.info(f"========== Training Pipeline Completed | Model Score: {acc_score} ==========")
        print(f"✅ Model Training Completed | Score: {acc_score}")

    except Exception as e:
        logging.error("error has occurred")
        raise CustomException(e,sys)
        



