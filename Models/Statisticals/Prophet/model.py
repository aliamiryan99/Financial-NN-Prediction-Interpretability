# Prophet_Model.py

import numpy as np
import pandas as pd
from prophet import Prophet
from Models.model_base import ModelBase
import warnings

from Configs.config_schema import Config

warnings.filterwarnings("ignore")  # Suppress warnings

# ============================
# Global Prophet Parameters
# ============================

PROPHET_GROWTH = 'linear'  # Can be 'linear' or 'logistic'
PROPHET_DAILY_SEASONALITY = True
PROPHET_WEEKLY_SEASONALITY = False
PROPHET_YEARLY_SEASONALITY = False
PROPHET_SEASONALITY_MODE = 'additive'  # Can be 'additive' or 'multiplicative'
PROPHET_CHANGEOINT_PRIOR_SCALE = 0.05
PROPHET_SEASONALITY_PRIOR_SCALE = 10.0
PROPHET_HOLIDAYS_PRIOR_SCALE = 10.0

# ============================

class ForecastingModel(ModelBase):
    def __init__(self, config: Config):
        super().__init__(config)
        self.model = None

    def build(self):
        """
        Build the Prophet model using the global parameters.
        """
        self.model = Prophet(
            growth=PROPHET_GROWTH,
            daily_seasonality=PROPHET_DAILY_SEASONALITY,
            weekly_seasonality=PROPHET_WEEKLY_SEASONALITY,
            yearly_seasonality=PROPHET_YEARLY_SEASONALITY,
            seasonality_mode=PROPHET_SEASONALITY_MODE,
            changepoint_prior_scale=PROPHET_CHANGEOINT_PRIOR_SCALE,
            seasonality_prior_scale=PROPHET_SEASONALITY_PRIOR_SCALE,
            holidays_prior_scale=PROPHET_HOLIDAYS_PRIOR_SCALE
        )

    def prepare_data(self, train, test):
        """
        Prepare the data for Prophet modeling.

        Parameters:
        - train (pd.DataFrame): The training dataset.
        - test (pd.DataFrame): The testing dataset.

        Returns:
        - X_train, y_train, X_test, y_test: Prepared data in the required format.
        """
        target_column = self.config.model_parameters.target_column

        # Remove timezone information from the index
        train.index = train.index.tz_localize(None)
        test.index = test.index.tz_localize(None)

        # For Prophet, we need 'ds' (datetime) and 'y' (target value)
        train_prepared = train.copy()
        train_prepared.rename(columns={target_column: 'y'}, inplace=True)
        train_prepared['ds'] = train_prepared.index

        test_prepared = test.copy()
        test_prepared.rename(columns={target_column: 'y'}, inplace=True)
        test_prepared['ds'] = test_prepared.index

        # X is not used in Prophet, but we still return an array for consistency
        X_train = np.array(train_prepared['ds'])
        y_train = np.array(train_prepared['y'])
        X_test = np.array(test_prepared['ds'])
        y_test = np.array(test_prepared['y'])

        return X_train, y_train, X_test, y_test

    def train(self, X_train, y_train):
        """
        Fit the Prophet model to the training data.

        Parameters:
        - X_train: Not used in Prophet.
        - y_train: Target values for training (a DataFrame in Prophet's case).
        """
        # Prophet requires a DataFrame with 'ds' and 'y' columns
        train_data = pd.DataFrame({'ds': X_train, 'y': y_train})
        self.model.fit(train_data)

    def forecast(self, X_test):
        """
        Make predictions using the trained Prophet model.

        Parameters:
        - X_test (np.ndarray): The test data (datetime values as 'ds').

        Returns:
        - y_pred (np.ndarray): Forecasted values.
        """
        future = pd.DataFrame({'ds': X_test})
        forecast = self.model.predict(future)
        y_pred = forecast['yhat'].values  # Extract the 'yhat' column as predictions

        return y_pred

def run(config: Config):
    """
    Orchestrate the Prophet modeling process using the base class's run method.
    """
    model = ForecastingModel(config)
    model.run()
