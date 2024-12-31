# Import Necessary Libraries
import warnings
from sklearn.ensemble import GradientBoostingRegressor  # Import Gradient Boosting Regressor

from Configs.config_schema import Config
from Modules.ModelModules.modules import create_sequences
from Models.model_base import ModelBase  # Assuming ModelBase is in Controllers.ModelBase

warnings.filterwarnings("ignore")  # Suppress warnings

class ForecastingModel(ModelBase):
    def __init__(self, config: Config):
        super().__init__(config)
        self.seq_length = config.model_parameters.seq_length
        self.num_features = len(config.model_parameters.feature_columns)
        # Hyperparameters for Gradient Boosting Regressor
        self.n_estimators = 100
        self.learning_rate = 0.1
        self.max_depth = 3

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
        """Build the Gradient Boosting model."""
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42
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
