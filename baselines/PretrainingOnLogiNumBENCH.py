import logging
import os
import random
import sys
import json
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from myutils import load_data_for_pretraining

import datasets
import numpy as np

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    default_data_collator,
    set_seed,
)
import torch

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        metadata={
            "help":
            "The name of the dataset to use (via myutils.py)."
        })
    max_seq_length: int = field(
        default=128,
        metadata={
            "help":
            ("The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.")
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={
            "help": "Overwrite the cached preprocessed datasets or not."
        })
    shuffle_seed: int = field(
        default=42,
        metadata={
            "help":
            "Random seed that will be used to shuffle the train dataset."
        })
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help":
            ("For debugging purposes or quicker training, truncate the number of training examples to this "
             "value if set.")
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help":
            ("For debugging purposes or quicker training, truncate the number of evaluation examples to this "
             "value if set.")
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help":
            ("For debugging purposes or quicker training, truncate the number of prediction examples to this "
             "value if set.")
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help":
            "Path to pretrained model or model identifier from huggingface.co/models"
        })
    config_name: Optional[str] = field(
        metadata={
            "help":
            "Pretrained config name or path if not the same as model_name"
        })
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help":
            "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    early_stopping_patience: int = field(
        default=3,
        metadata={
            "help": "Number of evaluation steps with no improvement after which training will be stopped."}
    )
    early_stopping_threshold: float = field(
        default=0.001,
        metadata={
            "help": "Minimum change in the monitored metric to qualify as an improvement."}
    )


def get_label_list(raw_dataset, split="train") -> List[str]:
    label_list = raw_dataset[split].unique("label")
    # we will treat the label list as a list of string instead of int, consistent with model.config.label2id
    label_list = [str(label) for label in label_list]
    return label_list


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # only one argument means the config is in a json file,
        # parse it to get arguments.
        try:
            model_args, data_args, training_args = parser.parse_dict(
                {k: v for d in json.load(open(sys.argv[1], 'r', encoding='utf-8')).values() for k, v in d.items()})
        except json.JSONDecodeError as e:
            print("JSONDecodeError:", e)
            sys.exit(1)
        except Exception as e:
            print("Error:", e)
            sys.exit(1)
        print(
            f"model_args: {model_args}\n\ndata_args: {data_args}\n\ntraining_args: {training_args}")
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        )

    # Ensure data is mixed loginumbench
    assert data_args.dataset_name == "Mloginumbench", f"Only Mloginumbench dataset is supported, but got {data_args.dataset_name}"

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    training_args.logging_dir = os.path.join(
        training_args.logging_dir, f'{training_args.run_name}-{current_time}')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        +
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = load_data_for_pretraining("Mloginumbench")

    logger.info(f"Dataset loaded: {raw_datasets}")
    logger.info(raw_datasets)

    for split in ["train", "validation", "test"]:
        if split in raw_datasets:
            label_list = get_label_list(raw_datasets, split=split)
            diff = set(label_list).difference(set(["0", "1"]))
            if len(diff) > 0:
                logger.error(
                    f"Labels {diff} in {split} set: {label_list}, exit with code 1."
                )
                exit(1)

    label_list.sort()
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="text-classification",
        cache_dir="./cache",
        trust_remote_code=True,
    )

    config.problem_type = "single_label_classification"
    logger.info("setting problem type to single label classification")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.config_name,
        cache_dir="./cache",
        use_fast=True,
        trust_remote_code=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir="./cache",
        trust_remote_code=True,
        ignore_mismatched_sizes=False,
    )

    if training_args.do_train:
        label_to_id = {v: i for i, v in enumerate(label_list)}

        model.config.label2id = label_to_id
        model.config.id2label = {
            id: label
            for label, id in label_to_id.items()
        }
    else:
        logger.info("using label infos in the model config")
        logger.info("label2id: {}".format(model.config.label2id))
        label_to_id = model.config.label2id

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        examples["sentence"] = ["" for _ in examples["facts_nl"]]
        for i in range(len(examples["sentence"])):
            examples["sentence"][i] = examples["facts_nl"][i] + \
                examples["rules_nl"][i] + examples["assertion_nl"][i]
        # Tokenize the texts
        result = tokenizer(examples["sentence"],
                           padding=False,
                           max_length=max_seq_length,
                           truncation=True)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[str(l)] if l != -1 else -1)
                               for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=16,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset.")
        train_dataset = raw_datasets["train"]
        train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset),
                                    data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            if "test" not in raw_datasets and "test_matched" not in raw_datasets:
                raise ValueError(
                    "--do_eval requires a validation or test dataset if validation is not defined."
                )
            else:
                logger.warning(
                    "Validation dataset not found. Falling back to test dataset for validation."
                )
                eval_dataset = raw_datasets["test"]
        else:
            eval_dataset = raw_datasets["validation"]

        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset),
                                   data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        # remove label column if it exists
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset),
                                      data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(
                range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(
                f"Sample {index} of the training set: {train_dataset[index]}.")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions,
                                               tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        accuracy = accuracy_score(y_true=p.label_ids, y_pred=preds)
        precision = precision_score(y_true=p.label_ids, y_pred=preds)
        recall = recall_score(y_true=p.label_ids, y_pred=preds)
        f1 = f1_score(y_true=p.label_ids, y_pred=preds)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    if training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer,
                                                pad_to_multiple_of=8)
    else:
        data_collator = None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience,
                                         early_stopping_threshold=training_args.early_stopping_threshold)],
    )

    # Training
    if training_args.do_train:
        try:
            train_result = trainer.train()
        except torch.cuda.OutOfMemoryError as e:
            print(f"{model_args.model_name_or_path}-{data_args.dataset_name}-{training_args.per_device_train_batch_size}显存不足错误已捕获：", e)
            torch.cuda.empty_cache()
        metrics = train_result.metrics
        max_train_samples = (data_args.max_train_samples
                             if data_args.max_train_samples is not None else
                             len(train_dataset))
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
            eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        if "label" in predict_dataset.features:
            predict_dataset = predict_dataset.remove_columns("label")
        predictions = trainer.predict(predict_dataset,
                                      metric_key_prefix="predict").predictions
        predictions = np.argmax(predictions, axis=1)
        output_predict_file = os.path.join(training_args.output_dir,
                                           "predict_results.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info("***** Predict results *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    item = label_list[item]
                    writer.write(f"{index}\t{item}\n")
        logger.info("Predict results saved at {}".format(output_predict_file))
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "text-classification"
    }

    trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
