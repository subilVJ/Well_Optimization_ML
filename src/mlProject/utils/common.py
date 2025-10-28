import os
import sys
from box.exceptions import BoxValueError
import yaml
import json
import joblib
from mlProject import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any

@ensure_annotations
def read_yaml(path_to_yaml:Path)->ConfigBox:

    '''
    Read yaml file and retur

    Args:
    path_to_yaml:str--> Path like input

    Raises:
    value error: if the file is empty 
    e: empty file

    Return:
    configbox: configBox type
    
    '''

    try:
        with open(path_to_yaml) as yaml_file:

            content=yaml.safe_load(yaml_file)
        logger.info(f"yaml file {yaml_file} loaded succesfully")
        return ConfigBox(content)

    except BoxValueError:
        ValueError("yaml file is empty")
    except  Exception as e:
        raise e
    

@ensure_annotations
def create_directories(path_to_directories:list,verbose=True):
    '''
     Create list of directories

     Args:
        Path_to_directories(list): list of path of directories
        ingnore_log(bool,ingnore): Ingnore if multiple directories is to be created. Default to False
     '''
    for path in path_to_directories:
        os.makedirs(path,exist_ok=True)
        if verbose:
             logger.info(f"created directories {path}")


@ensure_annotations
def save_json(path:Path,data:dict):
    '''
        save jason data
        Args:
            path: Path to jason file
            data: data to be saved in jason file
        
    '''
    with open(path,"w") as f:
        json.dump(data,f,indent=4)
    logger.info(f"json file saved at {path}")


@ensure_annotations
def load_jsonfile(path:Path)-> ConfigBox:
    with open(path) as f:
        content=json.load(f)
    logger.info(f"Json file loaded succesfully fom path {path}")
    return(ConfigBox(f))


@ensure_annotations
def save_bin(data:Any,path:Path):
    '''
    save binary file
    Args:
    path(Path): Path to binary file
    data(Any): Data to be saved as binary file
    
    '''
    joblib.dump(value=data,filename=path)
    logger.info(f"binary file saved{path}")

@ensure_annotations
def load_bin(path:Path)->Any:
    '''load binary data
    Args:
    path: Path to binary file

    Returns:
    Any object stored in the file
    
    
    '''
    data=joblib.load(path)
    logger.info(f"binary file loaded from {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

