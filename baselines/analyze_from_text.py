import json
import re
from myutils import all_dataset_name, load_data_by_name
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def clean_latex_text(text):
    text = re.sub(r'\\\((.*?)\\\)', r'\1', text)
    prev_text = ""
    while prev_text != text:
        prev_text = text
        text = re.sub(r'\\text\{([^{}]*)\}', r'\1', text)
        text = re.sub(r'\\boxed\{([^{}]*)\}', r'\1', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = text.replace('\\', '')

    text = re.sub(r'\boxed', '', text)
    text = re.sub(r'\bext\b', '', text)

    text = re.sub(r'\s+', ' ', text)

    text = text.replace('{', '').replace('}', '').replace('(', '').replace(')', '').replace(
        '[', '').replace(']', '').replace(' ', '').replace('\n', '').replace(':', '')
    text = text.strip()

    return text


def extract_labels(texts):
    pattern = re.compile(r"answer:\s*(true|false)", re.IGNORECASE | re.VERBOSE)
    labels = []
    for example in texts:
        match = pattern.search(example.lower().replace(" ", ""))
        if match and match.group(1) == "true":
            labels.append(1)
        elif match and match.group(1) == "false":
            labels.append(0)
        else:
            example = clean_latex_text(example)
            if 'answeristrue' in example.lower():
                labels.append(1)
            elif 'answerisfalse' in example.lower():
                labels.append(0)
            else:
                labels.append(-1)
    return labels


def get_pred_labels(file_path, dataset_name):
    """
    file are in json format and like 
    {
        "D1": [a list of texts],
        "D2": [a list of texts],
        ...
    }
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    dataset = data[dataset_name]
    return extract_labels(dataset)


def get_true_labels(dataset_name):
    """get the true labels by dataset name"""
    dataset = load_data_by_name(dataset_name)['test'].select(range(200))
    return dataset['label']


def compute_metrics(true_labels, pred_labels):
    cnt = 0
    total_length = len(true_labels)
    if len(pred_labels) != total_length:
        total_length = min(len(true_labels), len(pred_labels))
        true_labels = true_labels[:total_length]
        pred_labels = pred_labels[:total_length]
        logging.error(
            f"Length of true_labels and pred_labels are not equal, where true_labels: {len(true_labels)}, pred_labels: {len(pred_labels)}")
    for i in range(total_length):
        if pred_labels[i] == -1:
            pred_labels[i] = 1 - true_labels[i]
            cnt += 1
    logging.warning(f"Found {cnt} examples with no prediction")
    accuracy = accuracy_score(true_labels, pred_labels) * 100
    precision = precision_score(true_labels, pred_labels) * 100
    recall = recall_score(true_labels, pred_labels) * 100
    f1 = f1_score(true_labels, pred_labels) * 100
    return accuracy, precision, recall, f1


def log_header(msg, length=40):
    logging.info("="*length)
    logging.info(msg)
    logging.info("="*length)


def log_metrics(msg, accuracy, precision, recall, f1, length=40):
    logging.info("-"*length)
    logging.info(msg)
    logging.info("-"*length)
    logging.info(f"Accuracy: {accuracy:.2f}")
    logging.info(f"Precision: {precision:.2f}")
    logging.info(f"Recall: {recall:.2f}")
    logging.info(f"F1: {f1:.2f}")
    logging.info("-"*length)


def analyze_by_dataset(dataset_name, pred_files):
    """analyze the results of the dataset across different files"""
    log_header(f"Analyzing dataset {dataset_name}, pred_files: {pred_files}")
    true_labels = get_true_labels(dataset_name)
    for file in pred_files:
        pred_labels = get_pred_labels(file, dataset_name)
        accuracy, precision, recall, f1 = compute_metrics(
            true_labels, pred_labels)
        log_metrics(f"Analyzing file {file}", accuracy, precision, recall, f1)


def analyze_by_file(pred_file):
    """analyze the results of the file"""
    log_header(f"Analyzing file {pred_file}")
    for dataset_name in all_dataset_name:
        true_labels = get_true_labels(dataset_name)
        pred_labels = get_pred_labels(pred_file, dataset_name)
        accuracy, precision, recall, f1 = compute_metrics(
            true_labels, pred_labels)
        log_metrics(
            f"Analyzing dataset {dataset_name}", accuracy, precision, recall, f1)


if __name__ == "__main__":
    files = ["llms/glm-base.json", "llms/glm-chat.json"]
    for file in files:
        analyze_by_file(file)
