import os
import sys

from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,StandardScaler

from src.utils import save_object

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_data_path=os.path.join("artifacts","preprocessor.pkl")
class DataTransformation:
    def __init__(self):
        self.transform_config=DataTransformationConfig()
    def get_data_transformer_object(self):
        try:
            logging.info("find the cat_cols and num_cols")
            symptoms = ['fever', 'headache', 'nausea', 'vomiting', 'fatigue', 'joint_pain','skin_rash', 'cough', 'weight_loss', 'yellow_eyes']
                     
            pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent"))])

            logging.info(f"disease columns: {symptoms}")
            
            preprocessor=ColumnTransformer([("pipelines",pipeline,symptoms) ])
                
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
            logging.info("error occureed in preprocessor making")
    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("Read train and test data completed")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            target_col='disease'
            
            X_train_df=train_df.drop(target_col,axis=1)
            y_train_df=train_df[target_col]
            X_test_df=test_df.drop(target_col,axis=1)
            y_test_df=test_df[target_col]    

            symptoms = X_train_df.columns.values
           

            #le = LabelEncoder()
            #y_train_df = le.fit_transform(y_train_df)
            #y_test_df = le.transform(y_test_df)

            logging.info("Obtaining preprocessing object")
            preprocessor_obj=self.get_data_transformer_object()

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")
            input_train_arr=preprocessor_obj.fit_transform(X_train_df)
            input_test_arr=preprocessor_obj.transform(X_test_df)

            symptom_index = {symptom: idx for idx, symptom in enumerate(symptoms)} 

            train_arr=np.c_[input_train_arr,np.array(y_train_df)]
            test_arr=np.c_[input_test_arr,np.array(y_test_df)]

            save_object(file_path=self.transform_config.preprocessor_data_path,obj=preprocessor_obj)

            return (train_arr,test_arr,symptom_index,self.transform_config.preprocessor_data_path)
        except Exception as e:
            raise CustomException(e,sys)
            logging.error("error occurred at data transformation")

