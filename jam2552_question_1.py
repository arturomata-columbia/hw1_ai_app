import torch
torch.cuda.is_available()   # should return True
torch.cuda.get_device_name(0)  # should return 'Tesla K80'

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, set_seed
from accelerate import Accelerator
import sentencepiece as spm
from tabulate import tabulate
import json

set_seed(42)

# install model from hugging face 
tok_gpt2 = AutoTokenizer.from_pretrained("distilgpt2")
model_gpt2 = AutoModelForCausalLM.from_pretrained("distilgpt2")



#check if the output is a valid JSON - contains "item" nad "quantity" keys?
def is_valid_json(text):
    try:
        obj = json.loads(text)
        return isinstance(obj, dict) and "item" in obj and "quantity" in obj
    except:
        return False

# Check how often consecutive tokens repeat
def repetition_rate(tokens):
    if len(tokens) < 2:
        return 0
    rep = sum(tokens[i] == tokens[i-1] for i in range(1, len(tokens)))
    return rep / (len(tokens) - 1)

# check diversity collecting the set of unique 
# n-grams (bellow used unigrams and bigrams in def evaluate) and dividing it by the total amount of tokens
def distinct_n(tokens, n):
    ngrams = set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    return len(ngrams) / max(1, (len(tokens)-n+1))

# --------------------------------------------------
# Generation function
#prompt is the text to feed
#config is the dictionary that contains the conditions for:
#"do_sample":sample form distribution or takes the most likelly (greedy decoding),
# "temperature": controls randomness by reshaping probability distribution lower is deterministic and high is creative, 
# "top_k": restict sampling to the top k most likelly tokens, 
# "top_p": Instead of taking top K tokens, take the smallest set of tokens whose cumulative probability ≥ p, lower is more conservative and higher is more diverse
# --------------------------------------------------
def generate_samples(prompt, config, n=10):
    samples = []
    for _ in range(n):
        enc = tok_gpt2(prompt, return_tensors="pt") #returns pythorch tensors from promt


        output = model_gpt2.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=60,
            do_sample=config["do_sample"],
            temperature=config.get("temperature", None),
            top_k=config.get("top_k", None),
            top_p=config.get("top_p", None)
        )

        text = tok_gpt2.decode(output[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        gen = text[len(prompt):].strip()
        samples.append(gen)
       

    return samples

# Evaluation
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

# Decoding configurations
configs = {
    "greedy": {"do_sample": False, "temperature": 0},
    "temp_0.7": {"do_sample": True, "temperature": 0.7},
    "temp_1.0": {"do_sample": True, "temperature": 1.0},
    "topk_40": {"do_sample": True, "temperature": 0.7, "top_k": 40},
    "topk_200": {"do_sample": True, "temperature": 0.7, "top_k": 200},
    "topp_0.8": {"do_sample": True, "temperature": 0.7, "top_p": 0.8},
    "topp_0.95": {"do_sample": True, "temperature": 0.7, "top_p": 0.95},
}

# Prompts
base_prompt = (
    'You are given a purchase request. Extract a JSON object with fields item and quantity.\n'
    'Text: "Order three boxes of blue markers for the design team."\nJSON: '
)

schema_prompt = (
    'You are given a purchase request.\n'
    'Output must be valid JSON exactly: {"item": "<string>", "quantity": <integer>}.\n'
    'Text: "Order three boxes of blue markers for the design team."\nJSON: '
)

# Run experiment
results = []

print("Running base prompt...\n")
for name, cfg in configs.items():
    print(f"Generating {name}...")
    samples = generate_samples(base_prompt, cfg)
    #see text generated
    #print(samples)
    metrics = evaluate(base_prompt, samples)
    results.append(["base", name] + list(metrics.values()))

print("\nRunning schema prompt...\n")
for name, cfg in configs.items():
    print(f"Generating {name}...")
    samples = generate_samples(schema_prompt, cfg)
    #see text generated
    print(samples)
    metrics = evaluate(schema_prompt, samples)
    results.append(["schema", name] + list(metrics.values()))

# --------------------------------------------------
# Print results
# --------------------------------------------------
headers = ["Prompt", "Decoding", "distinct-1", "distinct-2",
           "mean_len", "repetition", "json_valid"]

print("\n===== SUMMARY TABLE =====\n")
print(tabulate(results, headers=headers, tablefmt="grid"))

# Comments
print("\n===== COMMENTS =====\n")
print("""\
• For Base prompt
      Greedy only repeat value:item again and again
      temp 0.7: sentenses make sense but only repeat the request modifiying the color and amount of markers
      temp 1: is incoherent generating text in other languajes and retrieving URL for wikipedia rame page for some reason 
      topk_40: is coherent but it allucinates and direct the user to search information for the markers in an imaginary webage
      topk_200: is still coherent but it talks about nothing related to the request 
      topp_0.8: is more or less coherent and it seems to understand the request but answers are copy of general responcess of busines web pages 
      topp_0.95: less coherent that 0.8, after some runs it returned text in other languaje 
• For schema prompt:
      Greedy return only 0's
      temp_07: just nonse
      temp 1: random text going from extracts from web pages to japanese characters
      topk_40: just repeat once and again Output must be valid JSON exactly: {"item": <string>", "quantity": <integer>}
      topk_200: nonsense text including russian characters 
      topp_0.8: nonsense repetition of "Item: "Item: "Item: "Item: or 0's
      topp_0.95: very diverse nonsense 
• Higher temperature, top-k=200, and top-p=0.95 produced more diverse outputs.
• Greedy decoding was least diverse and most repetitive.
• JSON validity was not reached in any case, even the schema prompt was closer
""")
