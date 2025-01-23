import json
import random


def load_data(path):
    with open(path, 'r', encoding='utf-8') as jsf:
        data = json.load(jsf)
    return data


def write_data(data, path):
    with open(path, 'w', encoding='utf-8') as jsf:
        json.dump(data, jsf, ensure_ascii=False, indent=4)


def gene_count_data(data):
    return data | {
        "count": {
            key: len(value)
            for key, value in data.items() if isinstance(value, list)
        }
    }


if __name__ == '__main__':
    data = load_data('./configs/config.json')
    keys = ['entityPool', 'relationPool', 'attrPool']

    train_config, val_config, test_config = [{
        k: v
        for k, v in data.items() if k not in keys
    } for _ in range(3)]

    for k in keys:
        random.shuffle(data[k])
        total_len = len(data[k])
        train_max = int(total_len * 0.8)
        val_max = int(total_len * 0.9)
        train_config |= {k: data[k][:train_max]}
        val_config |= {k: data[k][train_max:val_max]}
        test_config |= {k: data[k][val_max:]}

    write_data(gene_count_data(train_config), './configs/train_config.json')
    write_data(gene_count_data(val_config), './configs/val_config.json')
    write_data(gene_count_data(test_config), './configs/test_config.json')
