from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from nltk.tokenize import sent_tokenize
import numpy as np
import nltk
import evaluate
from transformers import AutoModelForSeq2SeqLM
import pandas as pd
import torch
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict, load_from_disk
import os
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


args = parse_arguments(sys.argv)

os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
checkpoint = args['ckpt']
predata_file = args['dataFile']
prestore_file = args['storeFile']+'/tokenized_data'

tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def load_preprocess_data(data_file, store_file):
    raw = load_from_disk(data_file)
    raw = raw.map(lambda s: {'inputs': s["facts_nl"]+s["rules_nl"] +
                  s["assertion_nl"], 'targets': s['str_reason']+s['answer']})
    columns_to_keep = ["targets", 'inputs']
    raw = raw.remove_columns([col for col in list(
        raw['train'].features.keys()) if col not in columns_to_keep])

    tokenized_inputs = raw.map(lambda x: tokenizer(
        x["inputs"], truncation=True), batched=True, remove_columns=["inputs", "targets"])
    max_source_length = max([len(x)
                            for x in tokenized_inputs['train']["input_ids"]])
    tokenized_targets = raw.map(lambda x: tokenizer(
        x["targets"], truncation=True), batched=True, remove_columns=["inputs", "targets"])
    max_target_length = max([len(x)
                            for x in tokenized_targets['train']["input_ids"]])

    print(f"dataset size: {len(raw['train'])}")
    print(f"Max source length: {max_source_length}")
    print(f"Max target length: {max_target_length}")

    def preprocess_function(sample, padding="max_length"):
        model_inputs = tokenizer(
            sample["inputs"], max_length=max_source_length, padding=padding, truncation=True)
        labels = tokenizer(
            text_target=sample["targets"], max_length=max_target_length, padding=padding, truncation=True)

        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = raw.map(
        preprocess_function, batched=True, remove_columns=["inputs", "targets"])
    print(
        f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")
    tokenized_dataset.save_to_disk(store_file)


# train
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

nltk.download("punkt")
metric = evaluate.load("rouge")

# helper function to postprocess text


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(
        decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(
        pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result


# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)


# Define training args

def do_training(store_file, level):
    batchsz = int(args['batchsz'])
    training_args = Seq2SeqTrainingArguments(
        output_dir=store_file.replace("tokenized_data", "output"),
        per_device_train_batch_size=batchsz,
        per_device_eval_batch_size=batchsz,
        predict_with_generate=True,
        fp16=False,  # Overflows with fp16
        learning_rate=float(args['lr']),
        num_train_epochs=1000,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        # metric_for_best_model="overall_f1",
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5,  # Number of evaluations with no improvement before stopping
        early_stopping_threshold=0.0001,  # Minimum improvement in the monitored metric
    )

    tokenized_datasets = load_from_disk(store_file)
    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["dev"],
        tokenizer=tokenizer,
        callbacks=[early_stopping_callback],
        # compute_metrics=compute_metrics,
    )

    trainer.train()


for depth in range(6, 7):
    data_file = predata_file.format(depth=depth)
    store_file = prestore_file.format(depth=depth)
    load_preprocess_data(data_file, store_file)
    do_training(store_file, depth)

    data_file = data_file.replace(f"D{str(depth)}", f"LD{str(depth)}")
    store_file = store_file.replace(f"D{str(depth)}", f"LD{str(depth)}")
    load_preprocess_data(data_file, store_file)
    do_training(store_file, depth)
