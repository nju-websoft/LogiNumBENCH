from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
import signal
import threading
import uvicorn
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

app = FastAPI()


class GenerationRequest(BaseModel):
    system_instruction: str
    user_prompt: str


def generate_text_batch_instruction_model(requests: List[GenerationRequest]):
    texts = []
    for req in requests:
        messages = [
            {"role": "system", "content": req.system_instruction},
            {"role": "user", "content": req.user_prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        texts.append(text)

    model_inputs = tokenizer(texts, return_tensors="pt",
                             padding=True).to(model.device)

    outputs = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        pad_token_id=tokenizer.eos_token_id
    )

    input_ids = model_inputs["input_ids"]
    input_lengths = input_ids.shape[1]
    contains_input = [
        torch.equal(input_seq, output_seq[:input_lengths])
        for input_seq, output_seq in zip(input_ids, outputs)
    ]
    if all(contains_input):
        outputs_trimmed = outputs[:, input_lengths:]
    else:
        outputs_trimmed = outputs

    generated_texts = tokenizer.batch_decode(
        outputs_trimmed, skip_special_tokens=True)
    return generated_texts


def generate_text_batch_base_model(requests: List[GenerationRequest]):
    texts = []
    for req in requests:
        input_text = req.system_instruction + "\n" + req.user_prompt
        texts.append(input_text)

    model_inputs = tokenizer(texts, return_tensors="pt",
                             padding=True).to(model.device)

    outputs = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        pad_token_id=tokenizer.pad_token_id
    )

    input_ids = model_inputs["input_ids"]
    input_lengths = input_ids.shape[1]
    contains_input = [
        torch.equal(input_seq, output_seq[:input_lengths])
        for input_seq, output_seq in zip(input_ids, outputs)
    ]
    if all(contains_input):
        outputs_trimmed = outputs[:, input_lengths:]
    else:
        outputs_trimmed = outputs

    generated_texts = tokenizer.batch_decode(
        outputs_trimmed, skip_special_tokens=True)
    return generated_texts


@app.post("/generate")
def generate_text_batch(requests: List[GenerationRequest]):
    if model_type == "it" and "gemma" not in model_name:
        return generate_text_batch_instruction_model(requests)
    else:
        return generate_text_batch_base_model(requests)


@app.post("/shutdown")
def shutdown():
    def shutdown_server():
        server.should_exit = True
    threading.Thread(target=shutdown_server).start()
    return {"message": "Shutting down..."}


@app.post("/info")
def get_info():
    return {
        "model_type": model_type,
        "model_path": args.model,
        "port": args.port
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the LLM server.")
    parser.add_argument('--model', type=str, default='gpt2',
                        help='Path to the local LLM model.')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to run the server on.')
    parser.add_argument('--model_type', type=str,
                        choices=['it', 'base'], help='Type of the model: instruction or base.')
    args = parser.parse_args()
    model_name = args.model

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, device_map='auto', trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=config, device_map='auto', trust_remote_code=True)
    if args.model_type:
        model_type = args.model_type
    elif "base" in args.model:
        model_type = "base"
    elif any(x in args.model.lower() for x in ["it", "instruct", "-chat"]):
        model_type = "it"
    else:
        model_type = "base"

    config = uvicorn.Config(app, host="0.0.0.0",
                            port=args.port, log_level="info")
    server = uvicorn.Server(config)

    app.state.server = server

    server.run()
