from store_sales.logger import logging
from store_sales.exception import CustomException
from store_sales.entity.artifact_entity import *
from store_sales.entity.config_entity import *
from store_sales.constant import *
from store_sales.utils.utils import read_yaml_file
from store_sales.utils.utils import save_array_to_directory
from store_sales.utils.utils import save_data
from store_sales.utils.utils import save_object

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


import numpy as np
import pandas as pd
import os, sys


class Data_cleaning:

    def __init__(self,drop_columns, label_encoder_columns ):
        
        logging.info(f"\n{'*'*20} Data Cleaning Started {'*'*20}\n\n")
                                
        self.drop_columns = drop_columns
        self.label_encoder_columns = label_encoder_columns
        logging.info(f" Data Cleaning Pipeline Initiatiated ")
    
    def drop_columns(self,df: pd.DataFrame):
        try:
            fe_drop = ['year', 'month', 'week', 'quarter', 'day_of_week']
            
            columns_to_drop = [column for column in fe_drop if column in df.columns]
            columns_not_found = [column for column in fe_drop if column not in df.columns]

            if len(columns_not_found) > 0:
                logging.info(f"Columns not found: {columns_not_found}")
                return df
            else:
                logging.info(f"Dropping columns: {columns_to_drop}")
                df.drop(columns=columns_to_drop, axis=1, inplace=True)
            logging.info(f"Columns after dropping: {df.columns}")

            return df
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def date_datatype(self,df:pd.DataFrame):
        # Convert 'date' column to datetime
        df['date'] = pd.to_datetime(df['date'])

        return df
    

    def renaming_oil_price(self,df: pd.DataFrame):
        df = df.rename(columns={"dcoilwtico": "oil_price"})
        
        logging.info(" Oil Price column renamed ")
        
        return df
    
    def drop_null_unwanted_columns(self,df:pd.DataFrame):
        if 'id' in df.columns:
            logging.info("Dropping 'id' column...")
            df.drop(columns=['id'], inplace=True)
        else:
            logging.info("'id' column not found. Skipping dropping operation.")

        logging.info("Dropping rows with null values...")
        df.dropna(inplace=True)

        logging.info("Resetting DataFrame index...")
        df.reset_index(drop=True, inplace=True)

        logging.info("Columns dropped, null values removed, and index reset.")

        return df
    
    def handling_missing_values(self,df):
        logging.info("Handling missing values in 'transactions' column...")
        df['transactions'] = df['transactions'].fillna(df['transactions'].mean())

        logging.info("Checking 'oil_price' column for missing values...")
        missing_values = df['oil_price'].isna().sum()
        logging.info(f"Number of missing values: {missing_values}")

        logging.info("Forward-filling missing values in 'oil_price' column...")
        df['oil_price'].interpolate(method='linear', inplace=True)

         # Verify if missing values have been filled
        missing_values_after = df['oil_price'].isna().sum()
        logging.info(f"Number of missing values after filling: {missing_values_after}")

        columns_missing = ['holiday_type', 'locale', 'locale_name', 'description', 'transferred']

        for column in columns_missing:
            logging.info(f"Filling missing values in '{column}' column with mode...")
            mode_value = df[column].mode().iloc[0]
            df[column].fillna(mode_value, inplace=True)

        logging.info("Missing values handled.")
        
        return df
    
    def run_data_modification(self,df:pd.DataFrame):
          
        # Dropping Irrelevant Columns
        df= self.drop_columns(df)
        
        # Change Datatype of the column 
        df= self.date_datatype(df)
        
        # renaming Oil_Price
        df=self.renaming_oil_price(df)
        
        # dropping null values 
        df=self.drop_null_unwanted_columns(df)
        
        # handling missing Values 
        df=self.handling_missing_values(df)
        
        logging.info(f"top 5 columns of dataframe:- \n {df.head()}")
        return df


    def data_wrangling(self,df:pd.DataFrame):
        try:
            # Data Modification 
            data_modified=self.run_data_modification(df)
            
            logging.info(" Data Modification Done")
            logging.info("Column Data Types:")
            for column in data_modified.columns:
                logging.info(f"Column: '{column}': {data_modified[column].dtype}")
            return data_modified
        except Exception as e:
            raise CustomException(e,sys) from e
    
    def fit(self,X,y=None):
            return self
    
    def transform(self,X:pd.DataFrame,y=None):
        try:    
            data_modified = self.data_wrangling(X)
            col = self.col
            # Reindex the DataFrame columns according to the specified column sequence
            data_modified = data_modified.reindex(columns=col)
            #data_modified.to_csv("data_modified.csv", index=False)
            logging.info("Data Wrangling Done")
            arr = data_modified.values
                
            return arr
        except Exception as e:
            raise CustomException(e,sys) from e
    

class TimeDataTransformation:
    def __init__(self, time_data_transformation_config: TimeDataTransformationConfig,
                    data_ingestion_artifact: DataIngestionArtifact,
                    data_validation_artifact: DataValidationArtifact):
        try:
            logging.info(f"\n{'*'*20} Data Transformation log started {'*'*20}\n\n")
            self.time_data_transformation_config = time_data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
                        
            # Schema File path 
            self.schema_file_path = self.data_validation_artifact.schema_file_path
            
            # Reading data in Schema 
            self.schema = read_yaml_file(file_path=self.schema_file_path)

            self.label_encoder_columns = self.schema[LABEL_ENCODER_COLUMNS]
            self.columns_drop = self.schema[DROP_COLUMN_KEY]
            self.date_column=self.schema[DATE_COLUMN]
        except Exception as e:
            raise CustomException(e,sys) from e
        
    
    def get_feature_engineering_object(self):
        try:
            feature_engineering = Pipeline(steps = [("fe",Data_cleaning(
                drop_columns=self.drop_columns,
                label_encoder_columns= self.label_encoder_columns
            ))])
            return feature_engineering
        except Exception as e:
            raise CustomException(e,sys) from e
    
    def label_encodeing(self, df, label_encoder_columns):
        self.label_encoder_columns = label_encoder_columns
        le = LabelEncoder()
        for i in label_encoder_columns:
            df[i] = le.fit_transform(df[i])
        
        return df
    
