from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from transformers.generation.utils import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import os
import argparse

# 定义一个函数,用于解析由逗号分隔的GPU id列表


def gpu_list(string):
    gpu_ids = []
    for gpu_id in string.split(','):
        try:
            gpu_ids.append(int(gpu_id))
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid GPU id: {gpu_id}")
    return gpu_ids


# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='控制使用哪些GPU进行计算')
parser.add_argument('--gpu', type=gpu_list, default=[],
                    help='要使用的GPU id列表,用逗号分隔')

# 解析命令行参数
args = parser.parse_args()
print(','.join(map(str, args.gpu)))

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu))


PATH = '/model/Meta-Llama-3-8B-Instruct'


tokenizer = AutoTokenizer.from_pretrained(PATH)
model = AutoModelForCausalLM.from_pretrained(
    PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


def inference(sample):
    messages = [
        {"role": "system", "content": instr},
        {"role": "user", "content": sample['inputs']},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]

    return {"pred": tokenizer.decode(response, skip_special_tokens=True)}


if __name__ == "__main__":
    with open('../instr3.txt', 'r') as file:
        instr = file.read()
    data_path = '/logiNumBench/datas/'
    data_names = ["D1", "D2", "D3", "D4", "D5", "D6",
                  "LD1", "LD2", "LD3", "LD4", "LD5", "LD6"]
    for datan in data_names:
        datap = data_path+datan+'/disk'
        print('---------------------'+datan+'---------------------')
        test_datasets = load_from_disk(datap)['test'].select(range(200))
        with torch.no_grad():
            test_datasets = test_datasets.map(inference)

        df = pd.DataFrame(test_datasets)
        df.to_excel('./zero/llama3-'+datan+'.xlsx',
                    index=False, engine='xlsxwriter')