import os
import pandas as pd

initial_datasets_folder = os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "initial_files"))
initial_datasets = []
for filename in os.listdir(initial_datasets_folder):


complete_dataset = pd.concat(initial_datasets, axis=1)
complete_dataset.to_csv(, index=False)