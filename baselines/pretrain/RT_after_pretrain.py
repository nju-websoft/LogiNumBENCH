from transformers import Trainer, AdamW, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict, load_from_disk
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'
checkpoint = '/fine-tuneRT/largeF/pretrain/checkpoint-65000'
tknier_ckpt = '/model/bert-large-uncased'
predata_file = '/RuleTaker/mixedclean_data/disk'
prestore_file = '/fine-tuneRT/largeF/RT'
ckpt_file = '/fine-tuneRT/largeF/fromPT'
batchsz = 16
lr = 1e-5


def load_preprocess_data(data_file, store_file):
    raw = load_from_disk(data_file)
    raw = raw.map(lambda s: {'labels': 1 if s['answer'] == 'true' else 0})
    columns_to_keep = ["labels", 'inputs']
    raw = raw.remove_columns([col for col in list(
        raw['train'].features.keys()) if col not in columns_to_keep])

    tokenizer = AutoTokenizer.from_pretrained(tknier_ckpt)
    tokenized_inputs = raw.map(lambda x: tokenizer(
        x["inputs"], truncation=True), batched=True, remove_columns=["inputs", "labels"])
    max_source_length = max(len(x) for subset in [
                            'train', 'dev', 'test'] for x in tokenized_inputs[subset]['input_ids'])

    print(f"Max source length: {max_source_length}")

    def preprocess_function(sample):
        return tokenizer(sample["inputs"], max_length=min(512, max_source_length), padding="max_length", truncation=True)

    tokenized_dataset = raw.map(
        preprocess_function, batched=True, remove_columns=["inputs"])
    print(data_file, tokenized_dataset)

    tokenized_dataset.save_to_disk(store_file)


def do_training(store_file, level):
    tokenizer = AutoTokenizer.from_pretrained(tknier_ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2)
    tokenized_datasets = load_from_disk(store_file)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=ckpt_file,
        per_device_train_batch_size=batchsz,
        per_device_eval_batch_size=batchsz,
        learning_rate=lr,
        num_train_epochs=1000,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
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
        early_stopping_patience=5,
        early_stopping_threshold=0.0001,
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


if __name__ == '__main__':
    # load_preprocess_data(predata_file, prestore_file)
    do_training(prestore_file, 5)
