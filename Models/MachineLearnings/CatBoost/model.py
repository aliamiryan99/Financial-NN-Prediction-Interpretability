# Import Necessary Libraries
import warnings
import numpy as np
from catboost import CatBoostRegressor  # Import CatBoost regressor

from Configs.config_schema import Config
from Modules.ModelModules.modules import create_sequences
from Models.model_base import ModelBase  # Assuming ModelBase is in Controllers.ModelBase

warnings.filterwarnings("ignore")  # Suppress warnings

# ============================
# Global CatBoost Parameters
# ============================
CATBOOST_ITERATIONS = 1000
CATBOOST_DEPTH = 6
CATBOOST_LEARNING_RATE = 0.1
CATBOOST_LOSS_FUNCTION = 'RMSE'
CATBOOST_EVAL_METRIC = 'RMSE'
CATBOOST_RANDOM_STATE = 42
CATBOOST_THREAD_COUNT = -1
CATBOOST_USE_GPU = False  # Set to True if GPU is available and desired
# ============================

class ForecastingModel(ModelBase):
    def __init__(self, config: Config):
        super().__init__(config)
        self.seq_length = config.model_parameters.seq_length
        self.num_features = len(config.model_parameters.feature_columns)

    def prepare_data(self, train, test):
        model_parameters = self.config.model_parameters
        # Prepare sequences
        X_train, y_train = create_sequences(
            train[model_parameters.feature_columns].values,
            train[model_parameters.target_column].values,
            model_parameters.seq_length,
            flatten=True
        )
        X_test, y_test = create_sequences(
            test[model_parameters.feature_columns].values,
            test[model_parameters.target_column].values,
            model_parameters.seq_length,
            flatten=True
        )
        return X_train, y_train, X_test, y_test

    def build(self):
        """Build the CatBoost model."""
        self.model = CatBoostRegressor(
            iterations=CATBOOST_ITERATIONS,
            depth=CATBOOST_DEPTH,
            learning_rate=CATBOOST_LEARNING_RATE,
            loss_function=CATBOOST_LOSS_FUNCTION,
            eval_metric=CATBOOST_EVAL_METRIC,
            random_seed=CATBOOST_RANDOM_STATE,
            thread_count=CATBOOST_THREAD_COUNT,
            verbose=False,  # Suppress training output
            task_type='GPU' if CATBOOST_USE_GPU else 'CPU'
        )
        return self.model

    def train(self, X_train, y_train):
        if self.model is None:
            raise Exception("Model needs to be built before training.")
        self.model.fit(X_train, y_train)

    def forecast(self, X_test):
        if self.model is None:
            raise Exception("Model needs to be built before forecasting.")
        y_pred = self.model.predict(X_test)
        return y_pred.reshape(-1, 1)


def run(config: Config):
    model = ForecastingModel(config)
    model.run()
