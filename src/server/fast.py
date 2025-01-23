from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoModelForSeq2SeqLM
import asyncio
import torch
from itertools import chain
import logging
import json

logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

generation_instruction_v2 = '''Your task is to rephrase the description provided above, enhancing its fluency and grammatical accuracy. Ensure the original meaning remains intact, and do not alter the key terms, especially adjectives and verbs used in the formal representation. Avoid replacing these words with synonyms or introducing any auxiliary words that might modify their meaning.
Examples:
{}
'''.format('\n'.join([f"Formal representation: {e['fr']}\nNatural language: {e['nl']}\n{e['gene']}" for e in json.load(open('examples1.json', 'r', encoding='utf-8'))]))

print(generation_instruction_v2)

app = FastAPI()


class GLLM:
    def __init__(self, model_dir: str, gpu_id: int):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, device_map=None).half()
        self.model.to(f"cuda:{gpu_id}").eval()

    def chat(self, input_messages: list) -> list:
        with torch.no_grad():
            texts = [self.tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True) for msg in input_messages]
            model_inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=128
            )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            responses = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True)
            return responses


model_dir = "../../models/Qwen2.5-7B-Instruct"
llms = [GLLM(model_dir, i) for i in [0, 1, 2]]

chat_lock = asyncio.Lock()


async def process_sub_message(llm, input_data):
    if len(input_data) == 0:
        return []
    sub_msgs = [
        [
            {"role": "system", "content": generation_instruction_v2},
            {"role": "user", "content": "Formal representation: {}\nNatural language: {}".format(
                data['fr'], data['nl'])}
        ] for data in input_data
    ]
    genes = await asyncio.to_thread(llm.chat, sub_msgs)
    return [data | {"gene": gene} for data, gene in zip(input_data, genes)]


@app.post("/chat/")
async def chat(request: Request):
    user_data = await request.json()
    logger.info(f"Received request data: {len(user_data)}")

    async with chat_lock:
        chunk_size = len(user_data) // len(llms)
        sub_messages = [
            user_data[i * chunk_size:(i + 1) * chunk_size] for i in range(len(llms) - 1)
        ] + [user_data[(len(llms) - 1) * chunk_size:]]

        tasks = [process_sub_message(llm, sub_message)
                 for llm, sub_message in zip(llms, sub_messages)]
        gened_data = await asyncio.gather(*tasks)
        gened_data = list(chain.from_iterable(gened_data))

    return gened_data
