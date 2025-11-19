from dataclasses import dataclass
from pathlib import   Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_url: str
    local_data_file: Path
    unzip_file: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    train_data_path: Path
    test_data_path : Path
    train_model_data: Path
    test_model_data: Path
    preprocessor_obj_file_path: Path

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name : str
    target_column: str


@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path : Path
    model_path: Path
    metric_file_name: Path
    target_column: str
    mlflow_url: str