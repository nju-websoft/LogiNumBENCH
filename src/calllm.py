import concurrent.futures
import json
import os.path
import requests
from tqdm import tqdm

url = "http://localhost:8080/chat"


def post_batch(data_list):
    response = requests.post(url, json=data_list)
    response_data = response.json()
    return response_data


def process_questions_batch(questions, batch_size):
    answers = []

    def process_batch(batch):
        try:
            return post_batch(batch)
        except Exception as e:
            print(f"An error occurred while processing batch: {e}")
            return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(process_batch, questions[i:i + batch_size])
                   for i in range(0, len(questions), batch_size)]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            answers.extend(future.result())

    return answers


def process_real_data(data_list, batch_size=50*3):
    id_to_item = {}
    questions = []
    unique_ids = []

    for data_idx, data in enumerate(data_list):
        metadata = data.get('metadata', {})
        facts = metadata.get('facts', [])
        rules = metadata.get('rules', [])

        for fact_idx, fact in enumerate(facts):
            unique_id = f"data{data_idx}_fact{fact_idx}"
            id_to_item[unique_id] = (data_idx, 'facts', fact_idx)
            questions.append(fact)
            unique_ids.append(unique_id)

        for rule_idx, rule in enumerate(rules):
            unique_id = f"data{data_idx}_rule{rule_idx}"
            id_to_item[unique_id] = (data_idx, 'rules', rule_idx)
            questions.append(rule)
            unique_ids.append(unique_id)

    all_results = process_questions_batch(questions, batch_size)

    for unique_id, answer in zip(unique_ids, all_results):
        data_idx, item_type, item_idx = id_to_item[unique_id]
        data_list[data_idx]['metadata'][item_type][item_idx] = answer

    def facts_nl(facts):
        ret = "Fact:\n"
        for i, fact in enumerate(facts):
            ret += str(i) + ". " + fact['gene'] + '\n'
        return ret

    def rules_nl(rules):
        ret = "Rule:\n"
        for i, rule in enumerate(rules):
            ret += str(i) + ". " + rule['gene'] + '\n'
        return ret

    for data in data_list:
        data['facts_nl'] = facts_nl(data['metadata']['facts'])
        data['rules_nl'] = rules_nl(data['metadata']['rules'])
    return data_list


def read_jsonl(path):
    data_list = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            data_list.append(data)
    return data_list


def write_jsonl(path, data_list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for data in data_list:
            line = json.dumps(data, ensure_ascii=False)
            f.write(line + '\n')


def main(input_file, output_file):
    datas = json.load(open(input_file))
    datas = datas
    results = json.load(open(output_file)) if os.path.exists(
        output_file) else {}
    for idx in range(len(datas)):
        datas[idx]['idx'] = str(idx)
    datas = list(filter(lambda x: x['idx'] not in results, datas))

    answers = process_questions_batch(datas)
    answers = dict([(x['idx'], x) for x in answers])

    results |= answers
    print(f"results: {len(results)}")
    with open(output_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    data_names = [prefix + str(i) for prefix in ['D', 'LD']
                  for i in range(1, 7)]
    data_types = [tp + "_config" for tp in ['train', 'val', 'test']]
    with tqdm(total=len(data_names), desc="Total Loop") as outer_bar:
        for data_name in data_names:
            for data_type in data_types:
                path = os.path.join(
                    './templated-data', data_name, data_type, 'datajs.jsonl')
                data_list = read_jsonl(path)
                data_list = process_real_data(data_list)
                write_jsonl(path.replace("templated-data",
                            "optimized-data"), data_list)
            outer_bar.update(1)
