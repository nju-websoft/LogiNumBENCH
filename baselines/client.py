import os  # 新增：用于检查文件是否存在
import requests
import json
import argparse
from tqdm import tqdm
from datasets import Dataset
from myutils import load_data_by_name, all_dataset_name
import sys


def load_datasets(dataname):
    text_columns = "facts_nl,rules_nl,assertion_nl"

    test_datasets = load_data_by_name(dataname)["test"].select(range(200))

    def preprocess_function(examples):
        text_column_names = text_columns.split(",")
        examples["user_prompt"] = examples[text_column_names[0]]
        for column in text_column_names[1:]:
            for i in range(len(examples[column])):
                examples["user_prompt"][i] += examples[column][i]
        return examples

    test_datasets = test_datasets.map(
        preprocess_function, num_proc=4, batched=True)

    test_datasets = test_datasets.remove_columns(
        [column for column in test_datasets.column_names if column not in ["user_prompt"]])

    return test_datasets


# 获取服务器信息
def get_server_info(server_url):
    url = f"{server_url}/info"
    response = requests.post(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"获取服务器信息失败，状态码: {response.status_code}")
        return None


# 批量请求生成文本
def generate_text_batch(server_url: str, dataset: Dataset, batch_size: int, instr: str, model_type: str):
    url = f"{server_url}/generate"
    headers = {"Content-Type": "application/json"}
    results = []

    for i in tqdm(range(0, len(dataset), batch_size), desc="处理批次", leave=False):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        if model_type == "it":
            batch_data = json.dumps([{
                "system_instruction": instr,
                "user_prompt": item["user_prompt"]
            } for item in batch])
        else:
            batch_data = json.dumps([{
                "system_instruction": instr,
                "user_prompt": item["user_prompt"]
            } for item in batch])
        response = requests.post(url, headers=headers, data=batch_data)
        if response.status_code == 200:
            batch_result = response.json()
            results.extend(batch_result)
        else:
            print(f"请求失败，状态码: {response.status_code}, 内容: {response.content}")
    return results


# 发送关闭信息
def send_shutdown(server_url):
    url = f"{server_url}/shutdown"
    response = requests.post(url)
    if response.status_code == 200:
        print("服务器已关闭")
    else:
        print(f"关闭请求失败，状态码: {response.status_code}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="客户端程序")
    parser.add_argument("--url", type=str, required=True, help="服务器的URL")
    parser.add_argument("--batch_size", type=int, default=2, help="批处理的大小")
    parser.add_argument("--output", type=str,
                        default="results.json", help="输出文件")
    args = parser.parse_args()

    server_url = args.url
    batch_size = args.batch_size
    output_file = args.output

    if 'it' in output_file or 'chat' in output_file:
        model_type = "it"
    elif 'base' in output_file:
        model_type = "base"
    else:
        sys.exit("无法识别的模型类型，程序终止。")

    # 获取服务器信息
    server_info = get_server_info(server_url)
    if server_info:
        print("服务器信息:", server_info)
    else:
        sys.exit("无法获取服务器信息，程序终止。")

    # 新增：加载已有结果文件（如果存在）
    if os.path.exists(output_file):
        with open(output_file, "r", encoding='utf-8') as f:
            try:
                existing_results = json.load(f)
                print(f"已加载现有结果文件: {output_file}")
            except json.JSONDecodeError:
                print(f"结果文件 {output_file} 无法解析为有效的JSON，将重新开始。")
                existing_results = {}
    else:
        existing_results = {}
        print(f"结果文件 {output_file} 不存在，将开始新处理。")

    # 确定需要处理的数据集
    datasets_to_process = []
    skipped_datasets = []

    for dataset_name in all_dataset_name:
        if dataset_name in existing_results:
            if len(existing_results[dataset_name]) == 200:
                skipped_datasets.append(dataset_name)
            else:
                print(
                    f"数据集 {dataset_name} 已存在但条目数不足 ({len(existing_results[dataset_name])}/200)，将重新处理。")
                datasets_to_process.append(dataset_name)
        else:
            datasets_to_process.append(dataset_name)

    print(f"跳过已完成的数据集 ({len(skipped_datasets)}): {skipped_datasets}")
    print(f"需要处理的数据集 ({len(datasets_to_process)}): {datasets_to_process}")

    results = existing_results  # 使用现有结果作为初始

    for idx, dataset_name in tqdm(enumerate(datasets_to_process), total=len(datasets_to_process), desc="处理数据集"):
        dataset = load_datasets(dataset_name)
        try:
            print(f"处理数据集 {idx+1}/{len(datasets_to_process)}: {dataset_name}")
            instr = open(f"instrs/gpt_prompt.txt", "r").read()
            if model_type == 'base':
                instr = instr + \
                    open(f"instrs/5-shot/{dataset_name}.txt", "r").read()
            print(f"指令: {instr}")
            result = generate_text_batch(
                server_url, dataset, batch_size, instr, model_type)
            if result:
                print(f"生成结果 for 数据集 {dataset_name}:")
                results[dataset_name] = result
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时出错: {e}")

    # 保存结果到输出文件
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"所有结果已保存到 {output_file}")

    send_shutdown(server_url)
