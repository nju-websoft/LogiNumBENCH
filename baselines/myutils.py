import os
from datasets import load_dataset, load_from_disk

all_dataset_name = [prefix + str(i)
                    for prefix in ['D', 'LD'] for i in range(1, 7)]


def load_data_by_name(dataset, base_dir='../data/loginumbench'):
    disk_path = os.path.join(f'{base_dir}-disk', dataset)
    if not os.path.exists(disk_path):
        dataset_path = os.path.join(base_dir, dataset)
        data_files = {
            "train": os.path.join(dataset_path, "train.jsonl"),
            "validation": os.path.join(dataset_path, "validation.jsonl"),
            "test": os.path.join(dataset_path, "test.jsonl"),
        }
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            cache_dir='./cache'
        )
        raw_datasets.save_to_disk(disk_path)
    else:
        raw_datasets = load_from_disk(disk_path)
    return raw_datasets


def load_data_for_pretraining(dataset):
    dataset2dir = {
        "mloginumbench": "../data/mixed_loginumbench",
        "ruletaker": "../data/RuleTaker"
    }
    dataset = dataset.lower()
    if dataset not in dataset2dir.keys():
        raise ValueError(f"dataset {dataset} not found")
    base_dir = dataset2dir[dataset]
    disk_path = f'{base_dir}-disk'
    if not os.path.exists(disk_path):
        data_files = {
            "train": os.path.join(base_dir, "train.jsonl"),
            "validation": os.path.join(base_dir, "validation.jsonl"),
            "test": os.path.join(base_dir, "test.jsonl"),
        }
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            cache_dir='./cache'
        )
        raw_datasets.save_to_disk(disk_path)
    else:
        raw_datasets = load_from_disk(disk_path)
    return raw_datasets
