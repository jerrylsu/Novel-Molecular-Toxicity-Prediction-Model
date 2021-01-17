from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Optional


class CSFPDataset(Dataset):
    """Dataset
    """
    def __init__(self, input_file_path: Optional[str], label_file_path: Optional[str]):
        super(CSFPDataset, self).__init__()
        self.features = pd.read_csv(input_file_path)
        self.labels = pd.read_csv(label_file_path)
        self.labels['label'] = self.labels['Toxicity'].apply(lambda x: 1 if x == 'P' else 0)
        self.labels = self.labels.drop(['Toxicity'], axis=1)
        self.dataset = self.features.set_index('Name').join(self.labels.set_index('Name'), how='inner')
        pass

    def __getitem__(self, item: Optional[int]):
        return self.dataset.iloc[item]

    def __len__(self) -> int:
        return self.dataset.shape[0]


if __name__ == '__main__':
    dataset = CSFPDataset('../..//data/train.csv', '../../data/Jiang1823Train.csv')
    print(len(dataset))
    pass
