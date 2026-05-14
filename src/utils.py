import os
import sys
import pickle

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


# Save object function
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


# Evaluate models
def evaluate_models(X_train, y_train, X_test, y_test, models, param):

    try:
        report = {}

        for model_name, model in models.items():

            para = param.get(model_name, {})

            # Apply GridSearchCV only if parameters exist
            if para:
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=para,
                    cv=3
                )

                gs.fit(X_train, y_train)

                model.set_params(**gs.best_params_)

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            # Scores
            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

            print(f"{model_name}")
            print(f"Train R2 Score: {train_model_score}")
            print(f"Test R2 Score: {test_model_score}")
            print("=" * 35)

        return report

    except Exception as e:
        raise CustomException(e, sys)


# Load object function
def load_object(file_path):

    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)