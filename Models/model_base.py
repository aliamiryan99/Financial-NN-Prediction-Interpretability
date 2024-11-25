# Import Necessary Libraries
import os
import pickle
import joblib
import matplotlib.pyplot as plt
import yaml  # Import for YAML serialization

from abc import ABC, abstractmethod
from Configs.config_schema import Config
from Controllers.ModelModules.modules import (preprocess_data, scale_data,
                                              split_data, create_sequences)
from Utils.io import load_data, save_forecasting_results
from tensorflow.keras.utils import plot_model

class ModelBase(ABC):
    def __init__(self, config: Config):
        self.config = config
        self.model = None  # Initialize model to None
        self.scalers = None # Initialize scaler to None

    @abstractmethod
    def build(self):
        """
        This method is responsible for building the model.
        Should be implemented by the derived class.
        """
        pass

    @abstractmethod
    def prepare_data(self, train, test):
        """
        This method is responsible for preparing and preprocessing the data.
        Should be implemented by the derived class.
        
        :param train: train input data to be prepared
        :param test: test input data to be prepared
        :return: Prepared and preprocessed data including X_train, y_tarin, X_test, y_test in numpy.ndarray format
        """
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        """
        This method is responsible for training the model.
        Should be implemented by the derived class.
        
        :param data: Training data
        """
        if self.model is None:
            raise Exception("Model needs to be built before training.")

    @abstractmethod
    def forecast(self):
        """
        This method is responsible for forecasting/predicting based on the model.
        Should be implemented by the derived class.
        
        :param data: Input data for forecasting
        :return: Forecasted result in numpy.ndarray format
        """
        if self.model is None:
            raise Exception("Model needs to be built before forcasting.")
    
    def inverse_transform(self, scaled_data, feature_name):
        """
        This method is responsible for reversing the scaling of predictions
        back to the raw data format.

        :param scaled_data: The scaled data (predictions) to inverse transform
        :param feature_name: The feature for which to apply the inverse transform
        :return: Data transformed back to its original scale
        """
        if self.scalers is None or feature_name not in self.scalers:
            raise Exception("Scalers must be set during data preparation before applying inverse transform.")

        return self.scalers[feature_name].inverse_transform(scaled_data)
        
    def run(self):
        model_parameters = self.config.model_parameters
        
        # Step 1: Load data
        print("Step 1: Loading the Data")
        data = load_data(self.config.data.in_path)

        # Step 2: Preprocess data
        print("Step 2: Preprocessing the Data")
        data = preprocess_data(data, model_parameters.feature_columns, filter_holidays=self.config.preprocess_parameters)

        # Step 3: Scale data
        print("Step 3: Scaling the Data")
        scaled_data, scalers = scale_data(data, model_parameters.feature_columns)
        self.scalers = scalers  # Save scalers for inverse_transform later

        # Step 4: Split data
        print("Step 4: Splitting the Data")
        train, test = split_data(scaled_data, model_parameters.train_ratio)

        # Step 5: Prepare the data
        print("Step 5: Preparing the Data")
        X_train, y_train, X_test, y_test = self.prepare_data(train, test)

        # Step 6: Build model
        print("Step 6: Building the Model")
        model = self.build()

        # Step 7: Train model
        print("Step 7: Training the Model")
        self.train(X_train, y_train)

        # Step 8: Forecast
        print("Step 8: Forecasting the Test Data")
        y_pred = self.forecast(X_test)
        y_pred = y_pred.reshape(-1, 1)

        # Step 9: Inverse transform the predictions and actual values
        print("Step 9: Inverse transforming test and predicted data")
        y_test = y_test.reshape(-1, 1)

        y_pred_inv = self.inverse_transform(y_pred, self.config.model_parameters.target_column)
        y_test_inv = self.inverse_transform(y_test, self.config.model_parameters.target_column)

        # Step 10: Save results
        save_forecasting_results(data, y_pred_inv.flatten(), self.config.data.out_path, model_parameters.target_column)

        # Define the experiments path
        experiments_path = self.config.data.exp_path
        if not os.path.exists(experiments_path):
            os.makedirs(experiments_path)

        # Step 11: Save the model, scaler, and loss plot
        print("Step 11: Saving the Model, Scaler, and Loss Plot, and Config")

        # Save the model
        model_save_path = os.path.join(experiments_path, 'model.keras')
        model.save(model_save_path)

        # Save the scaler
        scalers_save_path = os.path.join(experiments_path, 'scalers.pkl')
        joblib.dump(self.scalers, scalers_save_path)

        # Save the loss plot
        if hasattr(self, 'history'):
            plt.figure()
            plt.plot(self.history.history['loss'], label='Training Loss')
            if 'val_loss' in self.history.history:
                plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.legend()
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            loss_plot_path = os.path.join(experiments_path, 'loss_plot.png')
            plt.savefig(loss_plot_path)
            plt.close()
        
        # Save the config file
        config_save_path = os.path.join(experiments_path, 'config.pkl')
        with open(config_save_path, 'wb') as config_file:
            pickle.dump(self.config, config_file)

    
    
