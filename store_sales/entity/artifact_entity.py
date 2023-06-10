from collections import namedtuple

DataIngestionArtifact = namedtuple("DataIngestionArtifact",[
    "ingestion_file_path",
    "message"])

DataValidationArtifact = namedtuple("DataValidationArtifact",[
    "schema_file_path",
    "report_file_path",
    "report_page_file_path",
    "validated_file_path",
    "message"])

DataTransformationArtifact = namedtuple("DataTransformationArtifact",[
    "is_transformed",
    "message",
    "transformed_train_file_path",
    "transformed_test_file_path",
    "preprocessed_object_file_path",
    "feature_engineering_object_file_path"])




ModelTrainerArtifact = namedtuple("ModelTrainerArtifact",[
    "is_trained",
    "message",
    "trained_model_object_file_path"])

ModelEvaluationArtifact = namedtuple("ModelEvaluationArtifact",[
    "is_model_accepted",
    "improved_accuracy"])