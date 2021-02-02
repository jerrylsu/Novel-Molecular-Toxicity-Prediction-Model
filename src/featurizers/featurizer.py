from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from tqdm import tqdm
from typing import Optional, Tuple, Callable
from src.utils.utils import assert_statistics


class CSFPDataset(Dataset):
    """Dataset
    """
    def __init__(self, input_file: Optional[str]):
        super(CSFPDataset, self).__init__()
        self.dataset = torch.load(input_file)
        self.input_ids, self.labels = self.dataset["one_hots"], self.dataset["label"]
        pass

    def __getitem__(self, item: Optional[int]):
        input_ids, label = torch.tensor(self.input_ids[item]).float(), torch.tensor(self.label[item])
        return {"input_ids": input_ids, "label": label}

    def __len__(self) -> int:
        return len(self.labels)

    # def __init__(self, input_file_path: Optional[str], label_file_path: Optional[str]):
    #     super(CSFPDataset, self).__init__()
    #     self.features = pd.read_csv(input_file_path).set_index('Unnamed: 0')  # 1452
    #     self.labels = pd.read_csv(label_file_path).set_index('Name')    # 1458
    #     self.labels['label'] = self.labels['Toxicity'].apply(lambda x: 1 if x == 'P' else 0)
    #     self.labels = self.labels.drop(['Toxicity'], axis=1)
    #     self.dataset = self.features.join(self.labels, how='inner')

    #     assert_statistics(self.features, self.labels, self.dataset)
    #     pass

    # def __getitem__(self, item: Optional[int]):
    #     instance = self.dataset.iloc[item].to_list()
    #     input_ids, label = instance[:-1], instance[-1]
    #     input_ids, label = torch.tensor(input_ids).float(), torch.tensor(label)
    #     # ones_cnt = len(list(filter(lambda x: x == 1, instance[:-1])))
    #     return {"input_ids": input_ids, "label": label}

    # def __len__(self) -> int:
    #     return self.dataset.shape[0]


def get_dataloader(train_dataset: Optional[Dataset],
                   batch_size: Optional[int],
                   collate_fn: Optional[Callable],
                   validation_dataset: Optional[Dataset] = None,
                   shuffle: Optional[bool] = False,
                   num_workers: Optional[int] = 0) -> Tuple[DataLoader, Optional[DataLoader]]:
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=batch_size,
                                      collate_fn=collate_fn,
                                      shuffle=shuffle,
                                      num_workers=num_workers)
        validation_dataloader = DataLoader(dataset=validation_dataset,
                                           batch_size=batch_size,
                                           collate_fn=collate_fn,
                                           shuffle=shuffle,
                                           num_workers=num_workers)
        return train_dataloader, validation_dataloader


if __name__ == '__main__':
    validate_dataset = CSFPDataset("../../data/dataset/validate_file.pt")
    print(len(validate_dataset))
    train_dataloader, test_dataloader = get_dataloader(train_dataset=validate_dataset, batch_size=8)
    for batch in tqdm(train_dataloader, desc="Train_dataloader: "):
        batch = {key: value.to("cpu") for key, value in batch.items()}
        pass
