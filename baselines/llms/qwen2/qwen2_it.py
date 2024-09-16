from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from transformers.generation.utils import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import requests
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

PATH = 'Qwen2-7B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(PATH)
model = AutoModelForCausalLM.from_pretrained(
    PATH, device_map="auto", trust_remote_code=True).eval()


def inference(sample):
    messages = [
        {"role": "system", "content": instr},
        {"role": "user", "content": sample['inputs']}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=1024
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return {"pred": tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]}


if __name__ == "__main__":
    with open('../qwen2_prompt.txt', 'r') as file:
        instr = file.read()

    data_path = '/logiNumBench/datas/'
    data_names = ["D1", "D2", "D3", "D4", "D5", "D6",
                  "LD1", "LD2", "LD3", "LD4", "LD5", "LD6"]
    for datan in data_names:

        datap = data_path+datan+'/disk'
        print('---------------------'+datan+'---------------------')
        test_datasets = load_from_disk(datap)['test'].select(range(200))
        with torch.no_grad():
            test_datasets = test_datasets.map(inference)

        df = pd.DataFrame(test_datasets)
        df.to_excel('./zero/qwenchat-'+datan+'.xlsx',
                    index=False, engine='xlsxwriter')
