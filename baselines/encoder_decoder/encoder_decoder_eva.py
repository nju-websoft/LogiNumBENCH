import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import DatasetDict,load_dataset, load_from_disk
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import pandas as pd

def do_eva(workDr, tkdata, origdata, checkpoint, device='cuda:0'):

    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    model.to(device)

    tokenized_datasets = DatasetDict.load_from_disk(tkdata)['test']

    input_ids = torch.tensor(tokenized_datasets['input_ids'])
    attention_mask = torch.tensor(tokenized_datasets['attention_mask'])

    batchsz = 64
    dataset = TensorDataset(input_ids,attention_mask)
    dataloader = DataLoader(dataset, batch_size=batchsz)
    preds = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            outputs = model.generate(input_ids=batch[0].to(device),attention_mask=batch[1].to(device),max_length=512)
            preds.extend([tokenizer.decode(single) for single in outputs])

    text_datasets = load_from_disk(origdata)

    text_datasets = text_datasets['test']

    print(len(preds),len(text_datasets))


    df = pd.DataFrame.from_dict(text_datasets)
    df["prediction"] = preds
    return df

# modelPath, jsPath
prePath = './' # set the path by yourself
jsPath = '/logiNumBench/datas'

import json
with open('t5_config.json', 'r') as jsf:
    obj = json.load(jsf)

for modelName, mdict in obj.items():
    for dataName, ckpt in mdict.items():
        print(modelName, dataName)    
        workDr = prePath+'/'+modelName+'/'+dataName
        tkdata = workDr+'/'+'tokenized_data'
        checkpoint = workDr+'/output/checkpoint-'+str(ckpt)
        origdata = jsPath + '/' + dataName+'/disk'
        df = do_eva(workDr, tkdata, origdata, checkpoint, 'cuda:3')
        df.to_excel("./evas/"+modelName+"_"+dataName+".xlsx", index=False)
