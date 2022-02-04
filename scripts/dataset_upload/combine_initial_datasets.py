import os
import pandas as pd

initial_datasets_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "initial_files")
initial_datasets = []
for filename in os.listdir(initial_datasets_folder):
    initial_datasets.append(pd.read_csv(os.path.join(initial_datasets_folder, filename)))

complete_dataset = pd.concat(initial_datasets, axis=0)
complete_dataset.to_csv(os.path.join(initial_datasets_folder, "aggregated.csv"), index=False)