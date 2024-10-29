import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from detect_ai_content.ml_logic.registry import load_model

def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    if X_pred is None:
        X_pred = pd.DataFrame(
            dict(
                text_lenght=[2000],
                sentences_count=[23]
            )
        )

    model = load_model()
    assert model is not None

    X_processed = X_pred # preprocess_features(X_pred)
    y_pred = model.predict(X_processed)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    return y_pred

if __name__ == '__main__':
    pred()
