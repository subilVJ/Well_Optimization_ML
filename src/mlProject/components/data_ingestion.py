import os
import urllib.request as request
import zipfile
from mlProject import logger
from mlProject.utils.common import get_size
from pathlib import Path
from mlProject.entity.config_entity import DataIngestionConfig


class Dataingestion:

    def __init__(self,config:DataIngestionConfig):
        self.config=config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename,header=request.urlretrieve(
                url=self.config.source_url,
                filename=self.config.local_data_file
            )
            logger.info(f"{filename} downloaded! with folowing info :/n{header}")
        else:
            logger.info(f"file already exist of size {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        '''
        zip_file_path:str
        Extract the zip file into data directory
        Function returns none
        '''

        unzip_path=self.config.unzip_file
        os.makedirs(unzip_path,exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file,'r') as zip_ref:
            zip_ref.extractall(unzip_path)
