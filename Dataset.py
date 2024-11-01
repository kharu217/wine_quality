import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd

class wine_data(Dataset) :
    def __init__(self, input_path):
        csv_data = pd.read_csv(input_path)
        self.features = torch.tensor(csv_data.loc[:, :'alcohol'].values)
        self.quality = torch.tensor(csv_data.loc[:, 'quality'].values)
    
    def __getitem__(self, index):
        return self.features[index], self.quality[index]

    def __len__(self) :
        return len(self.quality)

if __name__ == '__main__' :
    path = 'Data\WineQT.csv'
    test_dataset = wine_data(path)
    print(test_dataset.quality[:10])
    print(test_dataset.features[:10])
