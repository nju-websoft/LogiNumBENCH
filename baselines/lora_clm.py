from datetime import datetime
import logging
import os
import random
import sys
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import datasets
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
import numpy as np
from myutils import load_data_by_name

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
    GenerationConfig
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
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum source sequence length after tokenization."}
    )
    max_target_length: Optional[int] = field(
        default=512,
        metadata={"help": "Maximum target sequence length after tokenization."}
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
    use_lora: bool = field(
        default=False,
        metadata={
            "help": "Whether to use lora model or not."
        }
    )
    lora_config: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Configuration for LoRA (Low-Rank Adaptation) if use_lora is True."
        }
    )
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

    raw_datasets = load_data_by_name(data_args.dataset_name)

    logger.info(f"Dataset loaded: {raw_datasets}")
    logger.info(raw_datasets)

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name else model_args.model_name_or_path,
        cache_dir='./cache',
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.config_name,
        cache_dir='./cache',
        use_fast=model_args.use_fast_tokenizer,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir='./cache',
        trust_remote_code=True,
        ignore_mismatched_sizes=False,
    )

    def model_to_lora_model(model):
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(**model_args.lora_config)
        model = get_peft_model(model, lora_config)
        return model
    if model_args.use_lora:
        model = model_to_lora_model(model)
        model.print_trainable_parameters()

    if data_args.max_source_length + data_args.max_target_length > tokenizer.model_max_length:
        logger.warning(
            "The sum of max_source_length and max_target_length is greater than the maximum sequence length for the model. "
            "This can cause errors in the tokenization process. "
            "Please reduce max_source_length and/or max_target_length to ensure the total sequence length is less than the maximum sequence length."
        )

    if data_args.max_source_length > tokenizer.model_max_length / 4 * 3:
        logger.warning(
            f"max_source_length {data_args.max_source_length} is greater than 3/4 of the maximum sequence length for the model {tokenizer.model_max_length}. "
            "This can cause errors in the tokenization process. "
            "Please reduce max_source_length to ensure the total sequence length is less than the maximum sequence length."
        )
    if data_args.max_target_length > tokenizer.model_max_length / 4:
        logger.warning(
            f"max_target_length {data_args.max_target_length} is greater than 1/4 of the maximum sequence length for the model {tokenizer.model_max_length}. "
            "Please reduce max_target_length to ensure the total sequence length is less than the maximum sequence length."
        )
    max_source_length = min(data_args.max_source_length,
                            tokenizer.model_max_length / 4 * 3)
    max_target_length = min(data_args.max_target_length,
                            tokenizer.model_max_length / 4)

    def preprocess_function(examples):
        examples["sentence"] = ["" for _ in examples["facts_nl"]]
        for i in range(len(examples["sentence"])):
            examples["sentence"][i] = examples["facts_nl"][i] + \
                examples["rules_nl"][i] + examples["assertion_nl"][i]

        examples["target"] = ["" for _ in examples["str_reason"]]
        for i in range(len(examples["target"])):
            examples["target"][i] = examples["str_reason"][i] + \
                examples["answer"][i]
        # Tokenize the inputs and targets
        model_inputs = tokenizer(
            examples["sentence"], max_length=max_source_length,
            padding=False, truncation=True, add_special_tokens=False, return_tensors=None
        )
        model_outputs = tokenizer(
            examples['target'], max_length=max_target_length,
            padding=False, truncation=True, add_special_tokens=False, return_tensors=None
        )

        input_ids = [i + o + [tokenizer.eos_token_id]
                     for i, o in zip(model_inputs.input_ids, model_outputs.input_ids)]
        attention_mask = [[1] * len(i) for i in input_ids]
        labels = [[-100] * len(i) + o + [tokenizer.eos_token_id]
                  for i, o in zip(model_inputs.input_ids, model_outputs.input_ids)]

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

    def data_collator_fn(features):
        batch_max_length = max([len(e['input_ids']) for e in features])

        input_ids = torch.tensor([[tokenizer.pad_token_id] * (
            batch_max_length - len(f['input_ids'])) + f['input_ids'] for f in features])
        attention_mask = torch.tensor(
            [[0] * (batch_max_length - len(f['attention_mask'])) + f['attention_mask'] for f in features])
        labels = torch.tensor(
            [[-100] * (batch_max_length - len(f['labels'])) + f['labels'] for f in features])

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

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
            logger.warning(
                f"Truncating training dataset to max_train_samples={data_args.max_train_samples}"
            )
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
            logger.warning(
                f"Truncating evaluation dataset to max_eval_samples={data_args.max_eval_samples}"
            )
            max_eval_samples = min(len(eval_dataset),
                                   data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        # remove label column if it exists
        if data_args.max_predict_samples is not None:
            logger.warning(
                f"Truncating prediction dataset to max_predict_samples={data_args.max_predict_samples}"
            )
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
        preds = p.predictions
        labels = p.label_ids

        preds = np.argmax(preds, axis=-1)
        mask = labels == -100
        labels[mask] = tokenizer.pad_token_id
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True)
        print(f"\n\ndecoded labels 0: {decoded_labels[0]}\n\n")
        # assert for all label, either true in label, or false in label
        assert all(any(result in label_text.lower().replace(" ", "").replace("\n", "") for result in [
                   'answer:true', 'answer:false']) for label_text in decoded_labels), "All labels must contain either 'Answer:\nTrue\n' or 'Answer:\nFalse\n'"

        preds = ['answer:true' in pred.lower().replace(" ", "").replace("\n", "")
                 for pred in decoded_preds]
        labels = ['answer:true' in label.lower().replace(
            " ", "").replace("\n", "") for label in decoded_labels]

        preds = np.array(preds).astype(int)
        labels = np.array(labels).astype(int)

        accuracy = accuracy_score(
            y_true=labels, y_pred=preds)
        precision = precision_score(
            y_true=labels, y_pred=preds, average='binary')
        recall = recall_score(
            y_true=labels, y_pred=preds, average='binary')
        f1 = f1_score(
            y_true=labels, y_pred=preds, average='binary')

        result = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        return result

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience,
                                         early_stopping_threshold=training_args.early_stopping_threshold)],
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
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

        predict_dataset = predict_dataset.remove_columns(
            column for column in predict_dataset.column_names
            if column not in ["input_ids", "attention_mask"])
        generation_config = GenerationConfig(
            max_new_tokens=data_args.max_target_length,
            temperature=0.7,
            top_p=0.8,
            do_sample=True
        )
        dataloader = torch.utils.data.DataLoader(
            predict_dataset, batch_size=training_args.per_device_eval_batch_size, collate_fn=data_collator_fn)

        predictions = []
        for batch in dataloader:
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=batch['input_ids'].to(
                        trainer.args.device),
                    attention_mask=batch['attention_mask'].to(
                        trainer.args.device),
                    **generation_config.to_dict()
                )
            decoded_preds = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True)
            predictions.extend(decoded_preds)

        output_predict_file = os.path.join(
            training_args.output_dir, "predict_results.npy")
        if trainer.is_world_process_zero():
            predictions_array = np.array(predictions)
            np.save(output_predict_file, predictions_array)
            logger.info(
                f"***** Predict results saved to {output_predict_file} *****")
        logger.info(f"Predict results saved at {output_predict_file}")

        # 创建模型卡
        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "tasks": "causal-lm-reasoning"
        }
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
