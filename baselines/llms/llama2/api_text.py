# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import uvicorn
import fire
from fastapi import FastAPI, Request
from llama import Llama
from typing import List
import json
import datetime
import os
generator = None
temperature = 0.6
top_p = 0.9
max_seq_len=8192
max_gen_len=1024
max_batch_size=1
ckpt_dir = '/llama/llama-2-7b'
tokenizer_path = '/llama/tokenizer.model'
os.environ["LOCAL_RANK"] = "0"

app = FastAPI()
@app.post("/")
async def create_item(request: Request):
    global generator, max_gen_len, temperature, top_p
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    prompts = [prompt]
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    response = results[0]['generation']
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    #torch_gc()
    return answer


def main():
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """ 
    global app, generator, max_seq_len, max_batch_size, tokenizer_path, ckpt_dir
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    uvicorn.run(app, host='0.0.0.0', port=32001, workers=1)


if __name__ == "__main__":
    fire.Fire(main)
