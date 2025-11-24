import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from accelerate import Accelerator
import sentencepiece as spm
from tabulate import tabulate
import json

# -----------------------
# Device setup
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -----------------------
# Reproducibility
# -----------------------
set_seed(42)

# -----------------------
# Load tokenizer & model
# -----------------------
tok_gpt2 = AutoTokenizer.from_pretrained("distilgpt2")
model_gpt2 = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)  # move to GPU

# -----------------------
# Helpers
# -----------------------
def is_valid_json(text):
    try:
        obj = json.loads(text)
        return isinstance(obj, dict) and "item" in obj and "quantity" in obj
    except:
        return False

def repetition_rate(tokens):
    if len(tokens) < 2:
        return 0
    rep = sum(tokens[i] == tokens[i-1] for i in range(1, len(tokens)))
    return rep / (len(tokens) - 1)

def distinct_n(tokens, n):
    ngrams = set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    return len(ngrams) / max(1, (len(tokens)-n+1))

# -----------------------
# Generation function (GPU-aware)
# -----------------------
def generate_samples(prompt, config, n=10):
    samples = []
    for _ in range(n):
        enc = tok_gpt2(prompt, return_tensors="pt").to(device)  # move input to GPU

        output = model_gpt2.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=60,
            do_sample=config["do_sample"],
            temperature=config.get("temperature", None),
            top_k=config.get("top_k", None),
            top_p=config.get("top_p", None)
        )

        text = tok_gpt2.decode(
            output[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        gen = text[len(prompt):].strip()
        samples.append(gen)
    return samples

# -----------------------
# Evaluation
# -----------------------
def evaluate(prompt, samples):
    d1_list, d2_list, rep_list, len_list = [], [], [], []
    valid = 0

    for s in samples:
        toks = tok_gpt2.encode(s)
        len_list.append(len(toks))
        d1_list.append(distinct_n(toks, 1))
        d2_list.append(distinct_n(toks, 2))
        rep_list.append(repetition_rate(toks))
        if is_valid_json(s):
            valid += 1

    return {
        "distinct-1": np.mean(d1_list),
        "distinct-2": np.mean(d2_list),
        "mean_len": np.mean(len_list),
        "repetition": np.mean(rep_list),
        "json_valid": valid / len(samples)
    }

# -----------------------
# Configs & prompts
# -----------------------
configs = {
    "greedy": {"do_sample": False, "temperature": 0},
    "temp_0.7": {"do_sample": True, "temperature": 0.7},
    "temp_1.0": {"do_sample": True, "temperature": 1.0},
    "topk_40": {"do_sample": True, "temperature": 0.7, "top_k": 40},
    "topk_200": {"do_sample": True, "temperature": 0.7, "top_k": 200},
    "topp_0.8": {"do_sample": True, "temperature": 0.7, "top_p": 0.8},
    "topp_0.95": {"do_sample": True, "temperature": 0.7, "top_p": 0.95},
}

base_prompt = (
    'You are given a purchase request. Extract a JSON object with fields item and quantity.\n'
    'Text: "Order three boxes of blue markers for the design team."\nJSON: '
)

schema_prompt = (
    'You are given a purchase request.\n'
    'Output must be valid JSON exactly: {"item": "<string>", "quantity": <integer>}.\n'
    'Text: "Order three boxes of blue markers for the design team."\nJSON: '
)

# -----------------------
# Run experiment
# -----------------------
results = []

print("Running base prompt...\n")
for name, cfg in configs.items():
    print(f"Generating {name}...")
    samples = generate_samples(base_prompt, cfg)
    metrics = evaluate(base_prompt, samples)
    results.append(["base", name] + list(metrics.values()))

print("\nRunning schema prompt...\n")
for name, cfg in configs.items():
    print(f"Generating {name}...")
    samples = generate_samples(schema_prompt, cfg)
    print(samples)
    metrics = evaluate(schema_prompt, samples)
    results.append(["schema", name] + list(metrics.values()))

# -----------------------
# Print summary table
# -----------------------
headers = ["Prompt", "Decoding", "distinct-1", "distinct-2",
           "mean_len", "repetition", "json_valid"]

print("\n===== SUMMARY TABLE =====\n")
print(tabulate(results, headers=headers, tablefmt="grid"))
