import os
from datetime import datetime

def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

CURRENT_TIME_STAMP = get_current_time_stamp()

ROOT_DIR = os.getcwd()  #to get current working directory
CONFIG_DIR = "config"
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(ROOT_DIR,CONFIG_DIR,CONFIG_FILE_NAME)

# Training pipeline related vaariables
TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline_config"
TRAINING_PIPELINE_ARTIFACT_DIR_KEY = "artifact_dir"
TRAINING_PIPELINE_NAME_KEY = "pipeline_name"



from store_sales.constant.training_pipeline import *

# Data_Ingestion related variables
# Data Ingestion related variable
DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATA_INGESTION_ARTIFACT_DIR = "data_ingestion"
DATA_INGESTION_RAW_DATA_DIR_KEY = "raw_data_dir"
DATA_INGESTION_INGESTED_DIR_NAME_KEY = "ingested_dir"
DATA_INGESTION_TRAIN_DIR_KEY = "ingested_train_dir"
DATA_INGESTION_TEST_DIR_KEY = "ingested_test_dir"
CONFIG_FILE_KEY = "config"


FILE_PATH="file_path"
FILE_NAME="file_name"
DESTINATION_FOLDER= r"C:\Users\Sumeet Maheshwari\Desktop\end to end project\Store Sales Forcasting using Time series\store-sales-forcasting\cleaned_data"


# Data Validation related variable
DATA_VALIDATION_CONFIG_KEY = "data_validation_config"
DATA_VALIDATION_ARTIFACT_DIR_NAME = "data_validation"
DATA_VALIDATION_SCHEMA_FILE_NAME_KEY = "schema_file_name"
DATA_VALIDATION_SCHEMA_DIR_KEY = "schema_dir"
DATA_VALIDATION_REPORT_FILE_NAME_KEY = "report_file_name"
DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY = "report_page_file_name"
DATA_VALIDATION_VALID_DATASET ="validated_data"


# Data Transformation related variables
PIKLE_FOLDER_NAME_KEY="Output_Folder"
# key  ---> config.yaml---->values
# Data Transformation related variables
DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation_config"
DATA_TRANSFORMATION_ARTIFACT_DIR = "data_transformation"
DATA_TRANSFORMATION_DIR_NAME_KEY = "transformed_dir"
DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY = "transformed_train_dir"
DATA_TRANSFORMATION_TEST_DIR_NAME_KEY = "transformed_test_dir"
DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY = "preprocessing_dir"
DATA_TRANSFORMATION_PREPROCESSING_FILE_NAME_KEY = "preprocessed_object_file_name"
DATA_TRANSFORMATION_FEATURE_ENGINEERING_FILE_NAME_KEY ="feature_engineering_object_file_name"

PIKLE_FOLDER_NAME_KEY = "Output_Folder"

# Model Training related variables
MODEL_TRAINER_CONFIG_KEY = "model_trainer_config"
MODEL_TRAINER_ARTIFACT_DIR = "model_training"
MODEL_TRAINER_TRAINED_MODEL_DIR = "trained_model_dir"
MODEL_TRAINER_TRAINED_MODEL_FILE_NAME_KEY = "model_file_name"

# Database related variables
DATABASE_CLIENT_URL_KEY = "mongodb://localhost:27017/?readPreference=primary&ssl=false&directConnection=true"
DATABASE_NAME_KEY = "Store_Sales_db"
DATABASE_COLLECTION_NAME_KEY = "Store_Sales"
DATABASE_TRAINING_COLLECTION_NAME_KEY = "Training"
DATABASE_TEST_COLLECTION_NAME_KEY = "Test"

