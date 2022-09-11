import os

import PIL.Image
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
import os
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection


class HsiDataset(Dataset):
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.csv_file_location = "data/out/hsi_csv.csv"
        self.work_csv_file_location = "data/out/hsi_work.csv"
        self.scalers = {}
        self.bands = list(range(1, 243))
        self.df = pd.read_csv(self.csv_file_location)
        train, test = model_selection.train_test_split(self.df, test_size=0.2)
        self.df = train
        if not self.is_train:
            self.df = test

        self.df = self._preprocess(self.df)
        self.df.to_csv(self.work_csv_file_location)

    def _preprocess(self, df):
        self.__scale__(df)
        return df

    def __scale__(self, df):
        self.__scale_col__(df, "soc")
        return df

    def __scale_col__(self, df, col):
        x = df[[col]].values.astype(float)
        self.scalers[col] = MinMaxScaler()
        x_scaled = self.scalers[col].fit_transform(x)
        df[col] = x_scaled
        return df

    def unscale(self, values, col):
        values = [[i] for i in values]
        values = self.scalers[col].inverse_transform(values)
        values = [i[0] for i in values]
        return values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        soc = self.df.iloc[idx]["soc"]
        reflectance = torch.tensor(self.df.iloc[idx,2:].values, dtype=torch.float32)
        return reflectance, torch.tensor(soc, dtype=torch.float32)


if __name__ == "__main__":
    cid = HsiDataset()
    dataloader = DataLoader(cid, batch_size=1, shuffle=True)
    for images, elevation, soc in dataloader:
        print(images)
        print(elevation)
        print(soc)
        exit(0)

