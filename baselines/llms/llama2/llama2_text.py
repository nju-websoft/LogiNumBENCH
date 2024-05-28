import requests
import json
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
import pandas as pd
import xlsxwriter
import torch


def get_answer(question, history=[{'role': 'system', 'content': instr}], url='http://127.0.0.0:32000'):
    '''
    :param question: 提问的问题(prompt) str
    :param history: 历史记录 [{'role':'user','content':'xxx'},{'role':'assistant','content':'yyy'},{'role':'system','content':'zzz'}]
    :param url: 接口地址
    :return: 回答结果 str
    '''
    headers = {
        'Content-Type': 'application/json',
    }
    data = {'history': history, 'prompt': question}
    while True:
        response = requests.post(url, headers=headers, json=data)
        # print(response)
        if response.status_code == 200:
            break
    # print(response.status_code)
    response_data = response.json()
    return response_data['response']


def inference(sample):
    return {"pred": get_answer(sample['inputs'], [{'role': 'system', 'content': instr + examples}])}


if __name__ == "__main__":
    with open('../llama_instr.txt', 'r') as file:
        instr = file.read()
    examples = ""
    data_path = '/logiNumBench/datas/'
    data_names = ["D1", "D2", "D3", "D4", "D5", "D6",
                  "LD1", "LD2", "LD3", "LD4", "LD5", "LD6"]
    for datan in data_names:
        with open("../shot-2/" + datan + ".txt", 'r') as file:
            examples = file.read()
        datap = data_path+datan+'/disk'
        print('---------------------'+datan+'---------------------')
        test_datasets = load_from_disk(datap)['test'].select(range(200))
        with torch.no_grad():
            test_datasets = test_datasets.map(inference)

        df = pd.DataFrame(test_datasets)
        df.to_excel('./fewshot-on-chat/llamafew-'+datan +
                    '.xlsx', index=False, engine='xlsxwriter')
