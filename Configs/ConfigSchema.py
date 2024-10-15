from dataclasses import dataclass
from typing import List

import yaml


@dataclass
class ModelParameters:
    feature_columns: List[str]
    target_column: str
    seq_length: int
    train_ratio: float
    epochs: int
    batch_size: int

@dataclass
class Data:
    name: str
    in_path: str
    out_path: str

@dataclass
class Config:
    model: str
    data: Data
    model_parameters: ModelParameters

# Utility function to load the YAML config
def load_config(config_path: str = "Configs/config.yaml") -> Config:
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
        model_params = ModelParameters(**config_dict['model_parameters'])
        data = Data(name=config_dict['data'], in_path=f"Data/{config_dict['data']}.csv",
                     out_path=f"Results/{config_dict['data']}/{config_dict['model']}.csv")
        return Config(
            model=config_dict['model'],
            data=data,
            model_parameters=model_params
        )