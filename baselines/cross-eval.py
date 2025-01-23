import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
from myutils import load_data_by_name, all_dataset_name
from tqdm import tqdm


def load_data(data_name, tokenizer, max_seq_length=512):
    def preprocess_function(examples):
        examples["sentence"] = ["" for _ in examples["facts_nl"]]
        for i in range(len(examples["sentence"])):
            examples["sentence"][i] = examples["facts_nl"][i] + \
                examples["rules_nl"][i] + examples["assertion_nl"][i]
        result = tokenizer(examples["sentence"],
                           padding=False,
                           max_length=max_seq_length,
                           truncation=True)
        return result | {"label": examples["label"]}
    raw = load_data_by_name(data_name)["test"]
    raw = raw.map(preprocess_function, batched=True, num_proc=16)
    raw = raw.remove_columns([col for col in raw.column_names if col not in [
                             "input_ids", "attention_mask", "label"]])
    return raw


def custom_collate_fn(batch, pad_token_id):
    input_ids = []
    attention_masks = []
    labels = []

    max_length = max(len(item['input_ids']) for item in batch)

    for item in batch:
        padding_length = max_length - len(item['input_ids'])

        padded_input_ids = item['input_ids'] + [pad_token_id] * padding_length
        input_ids.append(padded_input_ids)

        padded_attention_mask = item['attention_mask'] + [0] * padding_length
        attention_masks.append(padded_attention_mask)

        labels.append(item['label'])

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    labels = torch.tensor(labels)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'label': labels
    }


def eval_model_on_data(model_path, data_name, batch_size=128):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    test_data = load_data(data_name, tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=2).eval().cuda()
    dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, collate_fn=lambda x: custom_collate_fn(x, tokenizer.pad_token_id))
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch
            labels = inputs.pop("label")
            inputs = {key: value.to(model.device)
                      for key, value in inputs.items()}

            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    model_info = "-".join(model_path.split("/")[-1:-3:-1][::-1])
    directory = os.path.join("./cross-eval", model_info)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"{data_name}.txt"), "w") as f:
        for pred in all_preds:
            f.write(f"{pred}\n")

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


if __name__ == "__main__":
    path_dict = {
        "deberta-v3-large": {
            "D1": 5e-6,
            "D2": 5e-6,
            "D3": 5e-6,
            "D4": 5e-6,
            "D5": 1e-6,
            "D6": 5e-6,
            "LD1": 1e-5,
            "LD2": 5e-6,
            "LD3": 5e-6,
            "LD4": 5e-6,
            "LD5": 1e-6,
            "LD6": 5e-6
        },
        "bert-large-uncased": {
            "D1": 5e-6,
            "D2": 5e-6,
            "D3": 5e-6,
            "D4": 5e-6,
            "D5": 1e-6,
            "D6": 1e-6,
            "LD1": 1e-6,
            "LD2": 1e-6,
            "LD3": 5e-6,
            "LD4": 1e-5,
            "LD5": 1e-6,
            "LD6": 5e-6
        }
    }

    model_paths = []
    for model_name, dataset_lr in path_dict.items():
        for datan, lr in dataset_lr.items():
            model_paths.append(f"./output/{model_name}/{datan}_{lr}")

    # Create a DataFrame to store the results
    all_results = {}

    for model_path in tqdm(model_paths, desc="Evaluating models", leave=False):
        model_name = model_path.split("/")[-2]
        trained_dataname = model_path.split("/")[-1].split("_")[0]
        if model_name not in all_results:
            all_results[model_name] = {}

        # Initialize a list for the current model's results across datasets
        model_results = []
        for datan in tqdm(all_dataset_name, desc=f"Evaluating {model_path}"):
            accuracy = eval_model_on_data(model_path, datan)
            model_results.append(accuracy)
        all_results[model_name][trained_dataname] = model_results

with pd.ExcelWriter('cross_eval_results.xlsx') as writer:
    for model_name, results in all_results.items():
        # Create a DataFrame for the current model
        # Convert the results dict into a DataFrame where the rows are `trained_dataname`
        # and columns are `datan`
        df_model = pd.DataFrame(
            results, columns=all_dataset_name, index=results.keys())

        # Write this DataFrame to a sheet named after the model
        df_model.to_excel(writer, sheet_name=model_name)

print("Evaluation results saved to cross_eval_results.xlsx")
