import os
import pandas as pd

data_folder = os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data"))
pair_datasets = []
for filename in os.listdir(os.path.join(os.path.dirname(data_folder, "final_pair_datasets"))):


complete_dataset = pd.concat(pair_datasets, axis=1)
complete_dataset.to_csv(, index=False)