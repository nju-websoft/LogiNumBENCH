import sys
def parse_arguments(args):
    parsed_args = {}
    for arg in args[1:]:
        key_value = arg.split('=')
        if len(key_value) == 2:
            key = key_value[0]
            value = key_value[1]
            parsed_args[key] = value
        else:
            raise ValueError("args parsing fault {0}".format(arg))
    return parsed_args

# gpu, ckpt, dataFile, storeFile, batchsz, lr
# 获取命令行参数
args = parse_arguments(sys.argv)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer
import torch
import pandas as pd
checkpoint = args['ckpt']
predata_file = args['dataFile']
prestore_file = args['storeFile']+'/tokenized_data'

def load_preprocess_data(data_file, store_file):
    raw = load_from_disk(data_file)
    raw = raw.rename_column('label','labels')
    columns_to_keep = ["labels", 'inputs']
    raw = raw.remove_columns([col for col in list(raw['train'].features.keys()) if col not in columns_to_keep])

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenized_inputs = raw.map(lambda x: tokenizer(x["inputs"], truncation=True), batched=True, remove_columns=["inputs", "labels"])
    max_source_length = max(len(x) for subset in ['train', 'dev', 'test'] for x in tokenized_inputs[subset]['input_ids'])

    print(f"Max source length: {max_source_length}")

    def preprocess_function(sample):
        return tokenizer(sample["inputs"], max_length=min(512, max_source_length), padding="max_length", truncation=True)
        # if len(tokenized["input_ids"]) <= 512:
        #     return tokenized
        # else:
        #     return None

    tokenized_dataset = raw.map(preprocess_function, batched=True, remove_columns=["inputs"])
    print(data_file, tokenized_dataset)

    tokenized_dataset.save_to_disk(store_file)

from transformers import Trainer, AdamW, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, EarlyStoppingCallback
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support

def do_training(store_file, level):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    tokenized_datasets = load_from_disk(store_file)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=store_file.replace("tokenized_data", "output"),
        per_device_train_batch_size=int(args['batchsz']),  # 设置批次大小
        per_device_eval_batch_size=int(args['batchsz']),
        learning_rate=float(args['lr']),
        num_train_epochs=1000,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
        # evaluation_strategy="steps",  # 在每个步骤评估
        # eval_steps=100,  # 评估一次
        # save_steps=1,  # 保存模型
    )
    print("---Trainging Argument---")
    print(training_args)
    import evaluate
    metric = evaluate.load('accuracy')
    def compute_metrics(eval_preds):
        logits, true_labels = eval_preds
        pred = np.argmax(logits, axis=-1)
        return metric.compute(predictions=pred, references=true_labels)

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5,  # Number of evaluations with no improvement before stopping
        early_stopping_threshold=0.0001,  # Minimum improvement in the monitored metric
    )
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["dev"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )
    trainer.train()
    
    
for depth in range(1, 7):
    data_file = predata_file.format(depth=depth)
    store_file = prestore_file.format(depth=depth)
    load_preprocess_data(data_file, store_file)
    do_training(store_file, depth)
    
    data_file = data_file.replace(f"D{str(depth)}", f"LD{str(depth)}")
    store_file = store_file.replace(f"D{str(depth)}", f"LD{str(depth)}")
    load_preprocess_data(data_file, store_file)
    do_training(store_file, depth)
