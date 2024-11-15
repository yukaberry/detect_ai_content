import os
import pandas as pd

# Ensure the input and output paths point correctly to the file within the test_data folder
input_path = "texts_merged_dataset_enriched.csv"  # Directly use the file name if the script is in the same folder
output_path = "test_data.csv"  # Output will be saved in the same folder

# Load and clean the data
df = pd.read_csv(input_path)
df = df[['generated', 'text']]
df.to_csv(output_path, index=False)

print(f"Cleaned data saved to {output_path}")
