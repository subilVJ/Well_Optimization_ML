from mlProject import logger
from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_transformation import DataTransfromation
from pathlib import Path


STAGE_NAME="Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), 'r') as f:
                status=f.read().split(" ")[-1]

            if status=='True':
                config=ConfigurationManager()
                data_transfromation_config=config.get_data_transformation_config()
                data_transformation=DataTransfromation(config=data_transfromation_config)
                data_transformation.train_test_spliting()
                data_transformation.get_data_transformation()
                train_arr,test_arr,_=data_transformation.intiate_data_transfromation()

            else:
                raise Exception("Your data schema is not valid")
            
            return train_arr,test_arr
        
        except Exception as e:
            raise e
        

if __name__=="__main__":
    try:
        logger.info(f">>>>>>>>>> {STAGE_NAME} started <<<<<<<<<<<<<")
        obj=DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>>>{STAGE_NAME} completed <<<<<<<<<<")
    except Exception as e:
        raise e

