# Interpretations/Time/SHAP/model.py

import numpy as np
import shap
from Interpretations.Time.interpretation_base import InterpretationBase

class InterpretationModel(InterpretationBase):
    def interpret(self, X_test):
        """
        Uses SHAP's GradientExplainer (or other SHAP method) 
        to get SHAP values in shape (N, T, F).
        """
        # Ensure self.X_train is available
        if self.X_train is None:
            raise ValueError("No X_train found. Make sure 'run' sets self.X_train first.")

        # (1) Take some background data from X_train
        background = self.X_train[:500]  # e.g., subset of training data

        # (2) Build explainer
        print("Creating SHAP GradientExplainer...")
        explainer = shap.GradientExplainer(self.forecasting_model.model, background)

        # (3) Compute SHAP values => shape [N, T, F, 1] for a single-output model
        print("Computing SHAP values...")
        shap_values = explainer.shap_values(X_test)

        # Squeeze out the trailing singleton dimension => (N, T, F)
        shap_values_3d = np.squeeze(shap_values)

        # We can return raw shap values or absolute shap values. 
        # Usually, you'd want to do sum of absolute values, so let's just 
        # return shap_values_3d. The base class will handle absolute value 
        # and sums as needed.
        return shap_values_3d
