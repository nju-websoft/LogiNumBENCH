import torch
from transformers import  AutoModelForSequenceClassification, AutoTokenizer
from datasets import DatasetDict
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


def eva(checkpoint, tkdata, batchsz=128, device='cuda:2'):
    tokenized_datasets = DatasetDict.load_from_disk(tkdata)
    true_labels = [sample['labels'] for sample in tokenized_datasets["test"]]

    input_ids = torch.tensor(tokenized_datasets["test"]["input_ids"])
    attention_mask = torch.tensor(tokenized_datasets["test"]["attention_mask"])
    token_type_ids = torch.tensor(tokenized_datasets["test"]["token_type_ids"])
    tokenized_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

    pred = []
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    dataset = TensorDataset(input_ids,attention_mask,token_type_ids)
    dataloader = DataLoader(dataset, batch_size=batchsz)
    for batch in tqdm(dataloader):
        with torch.no_grad():
            outputs = model(input_ids=batch[0].to(device),attention_mask=batch[1].to(device),token_type_ids=batch[2].to(device))
            logits = outputs.logits
            t_pred = logits.argmax(dim=-1)
            pred.extend(t_pred.tolist())   

    accuracy = accuracy_score(true_labels, pred)

    precision = precision_score(true_labels, pred)
    
    recall = recall_score(true_labels, pred)

    f1 = f1_score(true_labels, pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
 
def eva_roberta(checkpoint, tkdata, batchsz=128, device='cuda:4'):
    tokenized_datasets = DatasetDict.load_from_disk(tkdata)
    true_labels = [sample['labels'] for sample in tokenized_datasets["test"]]
    input_ids = torch.tensor(tokenized_datasets["test"]["input_ids"])
    attention_mask = torch.tensor(tokenized_datasets["test"]["attention_mask"])
    tokenized_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    pred = []
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    dataset = TensorDataset(input_ids,attention_mask)
    dataloader = DataLoader(dataset, batch_size=batchsz)
    for batch in tqdm(dataloader):
        with torch.no_grad():
            outputs = model(input_ids=batch[0].to(device),attention_mask=batch[1].to(device))
            logits = outputs.logits
            t_pred = logits.argmax(dim=-1)
            pred.extend(t_pred.tolist())   

    accuracy = accuracy_score(true_labels, pred)

    precision = precision_score(true_labels, pred)

    recall = recall_score(true_labels, pred)

    f1 = f1_score(true_labels, pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

import json
with open("albert_xlarge_config.json", 'r') as jsf:
    obj = json.load(jsf)
    
dataNames = ["D1", "D2", "D3", "D4", "D5", "D6","LD1", "LD2", "LD3", "LD4", "LD5", "LD6"]
preModelPath = "./{modelName}/{dataName}/output/checkpoint-{ckpt}"
preDataPath = "./{modelName}/{dataName}/tokenized_data"
for modelName, mdict in obj.items():
    for trainData, sdict in mdict.items():
        print("-------------- {0} trained on {1} ------------------".format(modelName, trainData))

        modelPath = preModelPath.format(modelName=modelName, dataName=trainData, ckpt=sdict["output"])
        sdict["eva"] = []
        for testData in dataNames:
            res = eva_roberta(modelPath, preDataPath.format(modelName=modelName, dataName=testData))
            sdict["eva"].append(res)
        mdict[trainData] = sdict
    obj[modelName] = mdict
with open("albert_xlarge_res.json", 'w') as jsf:
    json.dump(obj, jsf)