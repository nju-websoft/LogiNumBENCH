"""
This script is an example of using the Zhipu API to create various interactions with a ChatGLM3 model. It includes
functions to:

1. Conduct a basic chat session, asking about weather conditions in multiple cities.
2. Initiate a simple chat in Chinese, asking the model to tell a short story.
3. Retrieve and print embeddings for a given text input.
Each function demonstrates a different aspect of the API's capabilities,
showcasing how to make requests and handle responses.

Note: Make sure your Zhipu API key is set as an environment
variable formate as xxx.xxx (just for check, not need a real key).
"""

from zhipuai import ZhipuAI
from datasets import load_from_disk
import pandas as pd
import xlsxwriter

base_url = "http://127.0.0.1:8000/v1/"
client = ZhipuAI(api_key="EMP.TY", base_url=base_url)


def simple_chat(messages, use_stream=True):
    response = client.chat.completions.create(
        model="chatglm3_",
        messages=messages,
        stream=use_stream,
        max_tokens=2048,
        temperature=0.8,
        top_p=0.8)
    if response:
        if use_stream:
            for chunk in response:
                print(chunk.choices[0].delta.content)
        else:
            content = response.choices[0].message.content
            return content
    else:
        print("Error:", response.status_code)


def embedding(inputs):
    response = client.embeddings.create(
        model="bge-large-zh-1.5",
        input=[inputs],
    )
    embeddings = response.data[0].embedding
    print("嵌入完成，维度：", len(embeddings))


def inference(sample):
    global instr, exmaples
    messages = [
        {
            "role": "system",
            "content": instr,
        },
        {
            "role": "user",
            "content": sample['inputs']
        }
    ]
    return {"pred": simple_chat(messages, False)}


if __name__ == "__main__":
    with open('../common_instr.txt', 'r') as file:
        instr = file.read()

    data_path = '/datas/'  # set by yourself
    data_names = ["D1", "D2", "D3", "D4", "D5", "D6",
                  "LD1", "LD2", "LD3", "LD4", "LD5", "LD6"]
    for datan in data_names:

        datap = data_path+datan+'/disk'
        print('---------------------'+datan+'---------------------')
        test_datasets = load_from_disk(datap)['test'].select(range(200))
        test_datasets = test_datasets.map(inference)

        df = pd.DataFrame(test_datasets)
        df.to_excel('./res/glm3-'+datan+'.xlsx',
                    index=False, engine='xlsxwriter')
