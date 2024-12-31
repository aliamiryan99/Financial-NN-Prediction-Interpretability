# Interpretations/Time/LIME/model.py

import numpy as np
from tqdm import tqdm
from lime.lime_tabular import LimeTabularExplainer
from Interpretations.Time.interpretation_base import InterpretationBase

class InterpretationModel(InterpretationBase):
    def interpret(self, X_test):
        """
        Return a 3D array (N, T, F) of LIME attributions.
        """
        num_samples, seq_length, num_features = X_test.shape

        # Flatten X_test for LIME
        X_test_flat = X_test.reshape(num_samples, -1)

        # Create feature names
        feature_names = [
            f"Feature_{f + 1}_Timestep_{t + 1}"
            for t in range(seq_length)
            for f in range(num_features)
        ]

        # Create the LimeTabularExplainer
        self.lime_explainer = LimeTabularExplainer(
            training_data=X_test_flat,
            mode="regression",
            feature_names=feature_names,
            class_names=["Prediction"]
        )

        # This will store (N, T, F)
        lime_importances_3d = np.zeros((num_samples, seq_length, num_features), dtype=np.float32)

        print("Interpreting sample predictions with LIME...")
        for i in tqdm(range(num_samples)):
            # We need a predict method that accepts flattened input
            def _predict_local(instance):
                # instance shape => (1, T*F)
                # reshape back to (1, T, F) for LSTM
                instance_reshaped = instance.reshape(-1, seq_length, num_features)
                return self.forecasting_model.model.predict(instance_reshaped)

            # Explain this sample
            explanation = self.lime_explainer.explain_instance(
                X_test_flat[i],
                _predict_local, 
                num_features=seq_length * num_features,
                num_samples=100
            )

            # LIME returns a list of (feature_index, weight) pairs in explanation.local_exp[0].
            # We'll fill them into a shape => (T, F)
            local_exp = dict(explanation.local_exp[0])  # dict: feature_idx -> weight

            # Now local_exp[k] is the importance weight for flatten-index k
            # We map k -> (t, f) and assign => lime_importances_3d[i, t, f]
            for k, weight in local_exp.items():
                # Flatten index k => t, f
                t = k // num_features
                f = k % num_features
                lime_importances_3d[i, t, f] = weight

        return lime_importances_3d
