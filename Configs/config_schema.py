from dataclasses import dataclass
from typing import List

import yaml


@dataclass
class ModelParameters:
    feature_columns: List[str]
    target_column: str
    seq_length: int
    train_ratio: float
    validation_split: float
    epochs: int
    batch_size: int
    optimizer: str
    loss: str
    verbose: bool

@dataclass
class PreprocessParameters:
    filter_holidays: bool

@dataclass
class Data:
    name: str
    in_path: str
    out_path: str
    exp_path: str
    interpret_path: str
    interpret_evaluation_path: str
    spec_interpret_path: str

@dataclass
class StreamVisualization:
    batch_size: int
    update_interval: int
    max_points: int
    time_frame: str
    show_aggregator: bool

@dataclass
class EvaluationVisualization:
    n_cols: int

@dataclass
class Config:
    model: str
    data: Data
    time_interpretability_class: str
    spectral_interpretability_class: str
    interpretability_type: str
    preprocess_parameters: PreprocessParameters
    model_parameters: ModelParameters
    stream_visualization: StreamVisualization
    evaluation_visualization: EvaluationVisualization

# Utility function to load the YAML config
def load_config(config_path: str = "Configs/config.yaml") -> Config:
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
        model_params = ModelParameters(**config_dict['model_parameters'])
        preprocess_params = PreprocessParameters(**config_dict['preprocess_parameters'])
        stream_visualization = StreamVisualization(**config_dict['stream_visualization'])
        evaluation_visualization = EvaluationVisualization(**config_dict['evaluation_visualization'])
        data = Data(name=config_dict['data'], in_path=f"Data/{config_dict['data']}.csv",
                     out_path=f"Results/Forecasting/{config_dict['data']}/{config_dict['model']}.csv",
                     exp_path=f"Exprements/{config_dict['data']}/{config_dict['model']}",
                     interpret_path=f"Results/Interpretability/{config_dict['data']}/{config_dict['model']}/{config_dict['time_interpretability_class']}.csv",
                     interpret_evaluation_path=f"Results/Interpretability/{config_dict['data']}/{config_dict['model']}/evaluations.csv",
                     spec_interpret_path=f"Results/Interpretability/{config_dict['data']}/{config_dict['model']}/Spectral/{config_dict['spectral_interpretability_class']}.csv")
        return Config(
            model=config_dict['model'],
            data=data,
            time_interpretability_class=config_dict['time_interpretability_class'],
            spectral_interpretability_class=config_dict['spectral_interpretability_class'],
            interpretability_type=config_dict['interpretability_type'],
            preprocess_parameters=preprocess_params,
            stream_visualization=stream_visualization,
            evaluation_visualization=evaluation_visualization,
            model_parameters=model_params
        )
    