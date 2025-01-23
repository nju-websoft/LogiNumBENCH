import concurrent
import json
import os.path
from myutils import load_data_by_name, all_dataset_name

from openai import OpenAI
from tqdm import tqdm

MAX_PROCESS_NUM = 200


def load_datasets(dataname):
    text_columns = "facts_nl,rules_nl,assertion_nl"

    test_datasets = load_data_by_name(
        dataname)["test"].select(range(MAX_PROCESS_NUM))

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


def call_api(model, client, data):
    id, msg = data
    response = client.chat.completions.create(
        model=model,
        messages=msg,
        stream=False
    )
    return (id, response)


def process_questions_multithreaded(model, client, concurrence, datas):
    answers = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrence) as executor:
        futures = [executor.submit(call_api, model, client, data)
                   for data in datas]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(datas)):
            try:
                answer = future.result()
                answers.append(answer)
            except Exception as e:
                print(f"An error occurred: {e}")

    return answers


def pack_message(instr, question):
    return [
        {"role": "system", "content": instr},
        {"role": "user", "content": question},
    ]


def restart_from_outputfile(output_file):
    if os.path.exists(output_file):
        with open(output_file, "r", encoding='utf-8') as f:
            try:
                existing_results = json.load(f)
                print(f"already load the existing results {output_file}")
            except json.JSONDecodeError:
                print(f"existing result {output_file} can't be parsed, will start new processing.")
                existing_results = {}
    else:
        existing_results = {}
        print(f"existing result {output_file} doesn't exist, will start new processing.")

    dataids_to_process = {}
    skipped_datasets = []

    for dataset_name in all_dataset_name:
        if dataset_name in existing_results:
            if len(existing_results[dataset_name]) == MAX_PROCESS_NUM:
                skipped_datasets.append(dataset_name)
            else:
                print(
                    f"result for dataset {dataset_name} already exist but doesn't enough ({len(existing_results[dataset_name])}/{MAX_PROCESS_NUM}), will start new processing.")
                dataids_to_process[dataset_name] = [str(i) for i in range(
                    MAX_PROCESS_NUM) if str(i) not in existing_results[dataset_name]]
        else:
            dataids_to_process[dataset_name] = [
                str(i) for i in range(MAX_PROCESS_NUM)]

    print(f"skip ({len(skipped_datasets)}): {skipped_datasets}")
    print(
        f"need to process {json.dumps({key: len(value) for key, value in dataids_to_process.items()}, indent=4)}")
    return existing_results, dataids_to_process


def normal_api_request(key, url, model, concurrence, zero_shot=True):
    if zero_shot:
        output_file = f"results/{model}-zero.raw.json"
    else:
        output_file = f"results/{model}-few.raw.json"
    print(key, url, model, output_file)
    client = OpenAI(api_key=key, base_url=url)
    instr = open(f"instrs/prompt.txt", "r").read()

    existing_results, dataids_to_process = restart_from_outputfile(output_file)
    datas_to_process = []

    for dataset_name in all_dataset_name:
        if dataset_name in dataids_to_process:
            dataset = load_datasets(dataset_name)
            if zero_shot:
                data_instr = instr
            else:
                data_instr = instr + \
                    open(f"instrs/5-shot/{dataset_name}.txt", "r").read()

            for dataid in dataids_to_process[dataset_name]:
                datas_to_process.append(
                    (f"{dataset_name}-{dataid}", pack_message(data_instr, dataset[int(dataid)]["user_prompt"])))

    raw_results = process_questions_multithreaded(
        model, client, concurrence, datas_to_process)

    for result_id, result_response in raw_results:
        dataset_name, dataid = tuple(result_id.split("-"))
        if dataset_name not in existing_results:
            existing_results[dataset_name] = {}
        existing_results[dataset_name][dataid] = result_response.dict()

    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(existing_results, f, ensure_ascii=False, indent=4)

    return output_file


def raw2text(raw_file):
    if not os.path.exists(raw_file):
        raise FileNotFoundError(f'{raw_file} not found')
    output_file = raw_file.replace('.raw.json', '.json')
    raw_results = json.load(open(raw_file, 'r', encoding='utf-8'))
    results = {}
    for dataset_name, responses in raw_results.items():
        results[dataset_name] = []
        for i in range(MAX_PROCESS_NUM):
            results[dataset_name].append(
                responses[str(i)]["choices"][0]["message"]["content"])
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    normal_api_request("api-xxx", "url-xxx", "model-xxx", zero_shot=True)
    normal_api_request("api-xxx", "url-xxx", "model-xxx", zero_shot=False)
