import re
import json
import numpy as np
import os
from datasets import load_dataset


def load_data_by_dir_and_name(dataset, base_dir='./optimized-data'):
    dataset_path = os.path.join(base_dir, dataset)
    data_files = {
        "train": os.path.join(dataset_path, "train_config", "datajs.jsonl"),
        "validation": os.path.join(dataset_path, "val_config", "datajs.jsonl"),
        "test": os.path.join(dataset_path, "test_config", "datajs.jsonl"),
    }
    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        cache_dir='./cache'
    )
    return raw_datasets


def fit_normal(data, desired_coverage=90):
    """
    Overlay a fitted normal distribution based on desired coverage, 
    and compute some statistical information. Ensures that median - 3σ >= 0.

    Parameters:
    - data: List or array of numerical values for analysis.
    - desired_coverage: Percentage of data to be covered within μ ± 3σ.
    """
    median = np.median(data)

    # Calculate σ based on desired coverage
    # Step 1: Compute absolute distances from the median
    distances = np.abs(np.array(data) - median)

    # Step 2: Find the distance threshold corresponding to the desired coverage percentile
    threshold = np.percentile(distances, desired_coverage)

    # Step 3: Calculate candidate σ
    sigma_candidate = threshold / 3

    # Step 4: Adjust σ to ensure median - 3σ >= 0
    if median - 3 * sigma_candidate >= 0:
        sigma = sigma_candidate
        adjusted = False
    else:
        sigma = median / 3
        adjusted = True
        # Recalculate the actual coverage with the adjusted σ
        actual_threshold = 3 * sigma
        actual_coverage = np.sum(
            distances <= actual_threshold) / len(distances) * 100

    statistics_info = {
        'median': median,
        'std_dev': sigma,
        'desired_coverage': desired_coverage,
        'threshold': threshold,
        'mu_minus_3sigma': max(median - 3*sigma, 0),
        'mu_plus_3sigma': median + 3*sigma,
    }
    if adjusted:
        statistics_info['adjusted'] = True
        statistics_info['actual_coverage'] = actual_coverage
    else:
        statistics_info['adjusted'] = False

    return statistics_info


def extract_last_number(samples, text='str_reason'):
    return {'number': [int(re.findall(r'\d+', x)[-1]) for x in samples[text]]}


def extract_last_number_float(text):
    # maybe float or int
    return float(re.findall(r'\d+\.\d+|\d+', text)[-1])


def modify_dataset(data):
    def replace_last_number(text, new_number):
        text = re.sub(r'\d+(?!.*\d)', str(new_number), text)
        return text
    data['assertion'] = replace_last_number(data['assertion'], data['sampled'])
    data['assertion_nl'] = replace_last_number(
        data['assertion_nl'], data['sampled'])
    assert extract_last_number_float(
        data['assertion']) == data['sampled'], f"Failed to replace last number in assertion: {data['assertion']}"
    assert extract_last_number_float(
        data['assertion_nl']) == data['sampled'], f"Failed to replace last number in assertion_nl: {data['assertion_nl']}"
    if 'greater' in data['assertion']:
        flag = True
    elif 'less' in data['assertion']:
        flag = False
    else:
        raise ValueError(f"Invalid assertion: {data['assertion']}")
    true_val = extract_last_number(data['str_reason'])
    if flag:
        data['label'] = true_val > data['sampled']
    else:
        data['label'] = true_val < data['sampled']
    data['answer'] = "Answer:\nTrue\n" if data['label'] else "Answer:\nFalse\n"
    data['answer_nl'] = "Answer:\nTrue\n" if data['label'] else "Answer:\nFalse\n"
    return data


def sample_from_normal(mu, sigma, num_samples=1000):
    """Sample from a normal distribution, ensuring all values are >= 1."""
    samples = np.random.normal(mu, sigma, num_samples)
    positive_samples = samples[samples >= 1]
    while len(positive_samples) < num_samples:
        additional_samples = np.random.normal(
            mu, sigma, num_samples - len(positive_samples))
        positive_samples = np.concatenate(
            (positive_samples, additional_samples[additional_samples > 1]))
    return positive_samples[:num_samples].round(2)


def read_statistics_from_json(report_file="statistics.json"):
    """Read statistics from a JSON file."""
    with open(report_file, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    all_dataset_name = [prefix + str(i)
                        for prefix in ['D', 'LD'] for i in range(1, 7)]
    DESIRED_COVERAGE = 90

    for dataset_name in all_dataset_name:
        data = load_data_by_dir_and_name(dataset_name)

        statistics = []
        extracted_data = data.map(
            extract_last_number, batched=True, num_proc=16)
        for split in extracted_data:
            statistics.extend(extracted_data[split]['number'])
        statistics = np.array(statistics)
        statistics_info = fit_normal(
            statistics, desired_coverage=DESIRED_COVERAGE)

        median = statistics_info['median']
        sigma = statistics_info['std_dev']

        for split in ['train', 'test', 'validation']:
            data[split] = data[split].add_column('sampled', sample_from_normal(
                median, sigma, num_samples=20000 if split == 'train' else 2000))

        data = data.map(modify_dataset, batched=True)
        data = data.remove_columns(['sampled', 'number'])

        for split in data:
            data[split].to_json(
                f"../data/loginumbench/{dataset_name}/{split}.jsonl", orient='records', lines=True)
