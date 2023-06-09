from collections import namedtuple
from datetime import datetime
import uuid
from threading import Thread
from typing import List
from store_sales.utils.utils import read_yaml_file
from multiprocessing import Process

from store_sales.config.Configuration  import Configuration
from store_sales.logger import logging
from store_sales.exception import ApplicationException
from store_sales.entity.artifact_entity import DataIngestionArtifact
from store_sales.entity.artifact_entity import DataValidationArtifact
from store_sales.entity.artifact_entity import DataTransformationArtifact
from store_sales.entity.artifact_entity import ModelTrainerArtifact
from store_sales.components.data_ingestion import DataIngestion
#from store_sales.components.data_validation import DataValidation
#from store_sales.components.data_transformation import DataTransformation
#from store_sales.components.Model_trainer import ModelTrainer
#from store_sales.constant import *



import os, sys
from collections import namedtuple
from datetime import datetime
import pandas as pd



class Pipeline():

    def __init__(self, config: Configuration = Configuration()) -> None:
        try:
            self.config = config
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def run_pipeline(self):
        try:
             #data ingestion

            data_ingestion_artifact = self.start_data_ingestion()
            #data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            #data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,
             #                                       data_validation_artifact=data_validation_artifact)
            #model_trainer_artifact = self.start_model_training(data_transformation_artifact=data_transformation_artifact)  

 
        except Exception as e:
            raise ApplicationException(e, sys) from e
