import argparse
import json
from chatgpt_request import QueueWithBar, ChatGPTThreadPool
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
import os
# parser = argparse.ArgumentParser()
# parser.add_argument('--input_file', type=str, required=True)
# args = parser.parse_args()

if __name__ == '__main__':
    with open('../instr3.txt', 'r') as file:
        instr = file.read()

    data_path = '/logiNumBench/datas/'
    data_names = ["D1", "D2", "D3", "D4", "D5", "D6",
                  "LD1", "LD2", "LD3", "LD4", "LD5", "LD6"]
    for datan in data_names:
        with open("../shot-2/" + datan + ".txt", 'r') as file:
            examples = file.read()
        datap = data_path+datan+'/disk'
        outfile = "./few/gpt_"+datan+".json"
        test_datasets = load_from_disk(datap)['test'].select(range(200))
        test_datas = {}
        if not os.path.isfile(outfile):
            with open(outfile, 'w', encoding='utf-8') as output_file:
                json.dump({}, output_file)

        with open(outfile, 'r', encoding='utf-8') as input_file:
            existing_data = json.load(input_file)
        for sample in test_datasets:
            if str(sample['id']) not in existing_data:
                test_datas[sample['id']] = {
                    'id': sample['id'], 'prompt': sample['inputs'], 'label': sample['label'], 'instr': instr + examples}
        if len(test_datas) == 200:
            exit()
        print(f'data count {datan}: {len(test_datas)}')

        tasks = QueueWithBar(total=len(list(test_datas)))
        for idx in list(test_datas):
            tasks.put(test_datas[idx])

        openai_api_keys = json.load(
            open('openai_api_keys.json', 'r', encoding='utf-8'))

        api_keys = openai_api_keys['api_keys']
        unavailable_keys = openai_api_keys['unavailable_keys']

        api_keys = list(filter(lambda x: x not in unavailable_keys, api_keys))

        print(len(api_keys))
        thread_pool = ChatGPTThreadPool(tasks=tasks, thread_num=len(
            api_keys), api_keys=api_keys, output_data_path=outfile)
        thread_pool.run()
