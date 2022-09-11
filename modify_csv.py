import pandas as pd

import torch

def modify():
    csv_file_location = "data/out/hsi_short.csv"
    work_csv_file_location = "data/out/hsi_modified.csv"
    df = pd.read_csv(csv_file_location)

    weights = torch.tensor([0.5, -0.3, 0.1, 0.7, -0.5])
    print(weights)

    for index, row in enumerate(df.iterrows()):
        reflectance = torch.tensor(row[1][2:7].values, dtype=torch.float32)
        x = torch.sum(reflectance*weights)
        df.at[index,"soc"] = x.numpy()

    df.to_csv(work_csv_file_location, index=False)


modify()