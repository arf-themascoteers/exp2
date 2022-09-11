import os
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from PIL import Image
import numpy
import pandas as pd

csv_file_location = "data/out/hsi_removed.csv"


def explore_sat():
    df = pd.read_csv(csv_file_location)
    array = df.iloc[0,2:].values
    plt.plot(array)
    plt.show()


explore_sat()
print("done")