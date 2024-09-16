from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk, DatasetDict

disk_folders = ['D1', 'D2', 'D3', 'D4', 'D5']
prepath = './datas/'
suffpath = '/disk'

train_datasets = []
dev_datasets = []
test_datasets = []

for disk_folder in disk_folders:
    dataset = load_from_disk(prepath + disk_folder + suffpath)
    train_datasets.append(dataset['train'])
    dev_datasets.append(dataset['test'])
    test_datasets.append(dataset['dev'])

train_data = concatenate_datasets(train_datasets)
dev_data = concatenate_datasets(dev_datasets)
test_data = concatenate_datasets(test_datasets)

train_data = train_data.shuffle(seed=63)
dev_data = dev_data.shuffle(seed=63)
test_data = test_data.shuffle(seed=63)

mix_data = DatasetDict({
    "train": train_data,
    "dev": dev_data,
    "test": test_data
})

print(mix_data)

mix_data.save_to_disk("./mix_disk")