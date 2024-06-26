from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from transformers.generation.utils import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import requests
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,4'


PATH = '/model/gemma-7b'

tokenizer = AutoTokenizer.from_pretrained(
    PATH, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    PATH, device_map="auto", trust_remote_code=True).eval()


def inference(sample):
    inputs = tokenizer([instr+examples+sample['inputs']],
                       return_tensors="pt").to('cuda')
    outputs = model.generate(**inputs, max_new_tokens=512)
    return {"pred": tokenizer.decode(outputs[0].tolist())}


if __name__ == "__main__":
    with open('../common_instr.txt', 'r') as file:
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
        df.to_excel('./res/gemmafew-'+datan+'.xlsx',
                    index=False, engine='xlsxwriter')
