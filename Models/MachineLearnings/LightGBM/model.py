# Import Necessary Libraries
import warnings
import numpy as np
from lightgbm import LGBMRegressor  # Import LightGBM regressor

from Configs.config_schema import Config
from Modules.ModelModules.modules import create_sequences
from Models.model_base import ModelBase  # Assuming ModelBase is in Controllers.ModelBase

warnings.filterwarnings("ignore")  # Suppress warnings

# ============================
# Global LightGBM Parameters
# ============================
LGBM_N_ESTIMATORS = 100
LGBM_MAX_DEPTH = 5
LGBM_LEARNING_RATE = 0.1
LGBM_OBJECTIVE = 'regression'
LGBM_N_JOBS = -1
LGBM_RANDOM_STATE = 42
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
        """Build the LightGBM model."""
        self.model = LGBMRegressor(
            n_estimators=LGBM_N_ESTIMATORS,
            max_depth=LGBM_MAX_DEPTH,
            learning_rate=LGBM_LEARNING_RATE,
            objective=LGBM_OBJECTIVE,
            n_jobs=LGBM_N_JOBS,
            random_state=LGBM_RANDOM_STATE
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
