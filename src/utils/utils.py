import pandas as pd
from collections import Mapping
import torch


def assert_statistics(features: pd.DataFrame, labels: pd.DataFrame, dataset: pd.DataFrame):
    print(f"Features shape: {features.shape}")  # 1452
    print(f"Labels shape: {labels.shape}")      # 1458
    feature_names = set(features.index)
    label_names = set(labels.index)
    difference_fl = label_names.difference(feature_names)
    print(f"Size: {len(difference_fl)}. The difference of label_names and feature_names: {difference_fl}.")
    intersection_fl = feature_names.intersection(label_names)
    print(f"Size: {len(intersection_fl)}. The intersection of feature_names and label_names: {intersection_fl}.")
    pass
    print(f"Dataset shape: {dataset.shape}")  # 1452
    dataset_names = set(dataset.index)
    intersection_fd = dataset_names.intersection(feature_names)
    print(f"Size: {len(intersection_fd)}. The intersection of dataset_names and feature_names: {intersection_fd}.")
    #assert dataset.shape[0] == features.shape[0]
    pass


def custom_collate_fn(batch):
    elem = batch[0]
    columns = ["input_ids", "label"]
    if isinstance(elem, Mapping):
        return {key: custom_collate_fn([instance[key] for instance in batch]) for key in elem}
    elif isinstance(elem, torch.Tensor):
        return torch.stack(batch, dim=0)
    else:
        return batch
