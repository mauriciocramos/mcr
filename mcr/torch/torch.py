import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np

class CustomImageDataset(Dataset):
    # source: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def create_sequences(data, seq_length, y_col_idx):
    """
    Generates input sequences and corresponding target values for time-series data.
    Parameters: input data, number of time steps in each sequence, target column name.
    Returns: NumPy arrays for input sequences an target values.
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
    # for i in range(len(data) // seq_length * seq_length - seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length, y_col_idx]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)