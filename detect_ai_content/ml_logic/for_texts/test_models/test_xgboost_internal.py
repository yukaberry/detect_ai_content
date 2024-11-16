import sys
import os
import pandas as pd

# Add the outer detect_ai_content directory to sys.path to access the ml_logic package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Import the LightGBM model class from lgbm_internal.py
from ml_logic.for_texts.xgboost_internal import XgBoostInternal

# Instantiate the model
xgboost_internal = XgBoostInternal()

# Define the correct path to the test data (assuming the test data is named 'test_data.csv')
test_data_path = os.path.join(os.path.dirname(__file__), "../test_data", "test_data.csv")

# Load the test data
try:
    test_data = pd.read_csv(test_data_path)
except FileNotFoundError:
    print(f"Error: '{test_data_path}' not found.")
    exit()

# Extract the text and labels for testing
texts = test_data['text']
true_labels = test_data['generated']

# Make predictions
predictions = [xgboost_internal.predict(text)[0] for text in texts]

# Evaluate predictions
accuracy = sum([1 for true, pred in zip(true_labels, predictions) if true == pred]) / len(true_labels)
print(f"Accuracy: {accuracy:.2%}")
