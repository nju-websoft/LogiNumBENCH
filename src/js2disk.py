from datasets import load_dataset, DatasetDict
data_file = "../datas/LD{level}/datajs.jsonl"
store_file = "../datas/LD{level}/disk"

for level in range(1,7):
    raw = load_dataset("json",data_files=data_file.format(level=level))
    raw = raw.map(lambda s: {'inputs': s["facts_nl"]+s["rules_nl"]+s["assertion_nl"], 'targets':s['str_reason']+s['answer']})

    raw_tmp = raw["train"].train_test_split(test_size=0.1)
    train_dev = raw_tmp['train'].train_test_split(test_size=1/9)
    raw = DatasetDict({
        "train": train_dev['train'],
        "dev": train_dev['test'],
        "test": raw_tmp['test']
    })
    print(raw)

    raw.save_to_disk(store_file.format(level=level))