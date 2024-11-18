import sys
import os
import pandas as pd

# Add the outer detect_ai_content directory to sys.path to access the ml_logic package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Import the RandomForest model class from rf_external.py
from ml_logic.for_texts.rf_external import RandomForestExternal  # Adjusted path to match the nested structure

rf_external = RandomForestExternal()

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
predictions = [rf_external.predict(text)[0] for text in texts]

# Evaluate predictions
accuracy = sum([1 for true, pred in zip(true_labels, predictions) if true == pred]) / len(true_labels)
print(f"Accuracy: {accuracy:.2%}")
