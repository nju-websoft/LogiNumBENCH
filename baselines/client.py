import os
import requests
import json
import argparse
from tqdm import tqdm
from datasets import Dataset
from myutils import load_data_by_name, all_dataset_name
import sys


def load_datasets(dataname):
    text_columns = "facts_nl,rules_nl,assertion_nl"

    test_datasets = load_data_by_name(dataname)["test"].select(range(200))

    def preprocess_function(examples):
        text_column_names = text_columns.split(",")
        examples["user_prompt"] = examples[text_column_names[0]]
        for column in text_column_names[1:]:
            for i in range(len(examples[column])):
                examples["user_prompt"][i] += examples[column][i]
        return examples

    test_datasets = test_datasets.map(
        preprocess_function, num_proc=4, batched=True)

    test_datasets = test_datasets.remove_columns(
        [column for column in test_datasets.column_names if column not in ["user_prompt"]])

    return test_datasets


def get_server_info(server_url):
    url = f"{server_url}/info"
    response = requests.post(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"fail to get server info, status code: {response.status_code}")
        return None


def generate_text_batch(server_url: str, dataset: Dataset, batch_size: int, instr: str, model_type: str):
    url = f"{server_url}/generate"
    headers = {"Content-Type": "application/json"}
    results = []

    for i in tqdm(range(0, len(dataset), batch_size), desc="generating with batch", leave=False):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        if model_type == "it":
            batch_data = json.dumps([{
                "system_instruction": instr,
                "user_prompt": item["user_prompt"]
            } for item in batch])
        else:
            batch_data = json.dumps([{
                "system_instruction": instr,
                "user_prompt": item["user_prompt"]
            } for item in batch])
        response = requests.post(url, headers=headers, data=batch_data)
        if response.status_code == 200:
            batch_result = response.json()
            results.extend(batch_result)
        else:
            print(f"fail to request, status code: {response.status_code}, content: {response.content}")
    return results


# send shutdown request to server
def send_shutdown(server_url):
    url = f"{server_url}/shutdown"
    response = requests.post(url)
    if response.status_code == 200:
        print("the server is shut down successfully.")
    else:
        print(f"fail to shutdown the server, status code: {response.status_code}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True, help="URL of the server")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--output", type=str,
                        default="results.json", help="output file")
    args = parser.parse_args()

    server_url = args.url
    batch_size = args.batch_size
    output_file = args.output

    if 'it' in output_file or 'chat' in output_file:
        model_type = "it"
    elif 'base' in output_file:
        model_type = "base"
    else:
        sys.exit("Can't determine model type from output file name, exit the program.")

    # get server info
    server_info = get_server_info(server_url)
    if server_info:
        print("server info: ", server_info)
    else:
        sys.exit("Can't get server info, exit the program.")

    # load existing results
    if os.path.exists(output_file):
        with open(output_file, "r", encoding='utf-8') as f:
            try:
                existing_results = json.load(f)
                print(f"already load existing results: {output_file}")
            except json.JSONDecodeError:
                print(f"existing result {output_file} can't be loaded, will start new processing.")
                existing_results = {}
    else:
        existing_results = {}
        print(f"existing result {output_file} doesn't exist, will start new processing.")

    datasets_to_process = []
    skipped_datasets = []

    for dataset_name in all_dataset_name:
        if dataset_name in existing_results:
            if len(existing_results[dataset_name]) == 200:
                skipped_datasets.append(dataset_name)
            else:
                print(
                    f"dataset {dataset_name} exist but not enough ({len(existing_results[dataset_name])}/200), will reprocess from start.")
                datasets_to_process.append(dataset_name)
        else:
            datasets_to_process.append(dataset_name)

    print(f"skip the completed dataset: ({len(skipped_datasets)}): {skipped_datasets}")
    print(f"need to process dataset: ({len(datasets_to_process)}): {datasets_to_process}")

    results = existing_results

    for idx, dataset_name in tqdm(enumerate(datasets_to_process), total=len(datasets_to_process), desc="dataset processing"):
        dataset = load_datasets(dataset_name)
        try:
            print(f"process dataset {idx+1}/{len(datasets_to_process)}: {dataset_name}")
            instr = open(f"instrs/prompt.txt", "r").read()
            if model_type == 'base':
                instr = instr + \
                    open(f"instrs/5-shot/{dataset_name}.txt", "r").read()
            print(f"instr: {instr}")
            result = generate_text_batch(
                server_url, dataset, batch_size, instr, model_type)
            if result:
                print(f"result for dataset {dataset_name}:")
                results[dataset_name] = result
        except Exception as e:
            print(f"fail when processing dataset {dataset_name}: {e}")

    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"results are stored in {output_file}")

    send_shutdown(server_url)
