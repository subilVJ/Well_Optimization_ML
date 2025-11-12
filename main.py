from mlProject import logger
from mlProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from mlProject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from mlProject.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline

STAGE_NAME='Data Ingestion Stage'
try:
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<")
    data_ingestion=DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<\n\nx===============x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME_1='Data Validation Satge'
try:
    logger.info(f">>>>> stage {STAGE_NAME_1}  started <<<<<")
    data_validation=DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f">>>> stage {STAGE_NAME_1} completed <<<<<")
except Exception as e:
    raise e

STAGE_NAME_2="Data Transformation Stage"

try:
    logger.info(f">>>>>>>>>> {STAGE_NAME_2} started <<<<<<<<<<<<<")
    data_transform=DataTransformationTrainingPipeline()
    train_arr,test_arr=data_transform.main()
    
    logger.info(f">>>>>>>>>{STAGE_NAME_2} completed <<<<<<<<<<")
except Exception as e:
    raise e






