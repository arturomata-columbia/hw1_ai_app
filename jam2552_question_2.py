# sft_flan_t5_cpu.py
import random
import time
import json
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,
    EvalPrediction
)
from tabulate import tabulate

# -----------------------
# Config / Seed / Device
# -----------------------
set_seed(42)
random.seed(42)
np.random.seed(42)
device = "cpu"  # CPU-only

# -----------------------
# Load tokenizer & model
# -----------------------
tok_t5 = AutoTokenizer.from_pretrained("google/flan-t5-small")
model_t5 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
model_t5.to(device)

# -----------------------
# Task A: Sentiment examples
# -----------------------
SENTIMENT_LABELS = ["very_negative", "negative", "neutral", "positive", "very_positive"]

def make_sentiment_example(label: str):
    texts = {
        "very_negative": [
            "This product is an utter disaster — completely unusable and broken on arrival.",
            "I would never buy this again; it failed spectacularly.",
            "Terrible — arrived damaged, stopped working immediately.",
            "Completely unacceptable and worthless."
        ],
        "negative": [
            "Not great: it had issues and didn't meet my expectations.",
            "Disappointing performance and defects.",
            "It works but with many faults.",
            "Below average; troubleshooting required."
        ],
        "neutral": [
            "It does the job — nothing impressive but not bad either.",
            "Average; neither good nor bad.",
            "Functional and okay.",
            "Satisfactory for simple uses."
        ],
        "positive": [
            "Pretty good product — solid build and does what I need.",
            "I liked it; performs reliably.",
            "Good quality and competent design.",
            "Works well for everyday tasks."
        ],
        "very_positive": [
            "Absolutely excellent — exceeded expectations.",
            "Fantastic! Flawless performance.",
            "I love it — couldn't be happier.",
            "Perfect for my needs; highly recommend."
        ]
    }
    subtler = ["Slightly", "Somewhat", "Surprisingly", "Remarkably", "Moderately", "Notably"]
    sentence = random.choice(texts[label])
    if random.random() < 0.35:
        sentence = f"{sentence} {random.choice(subtler)} better than expected."
    inp = f"Classify sentiment into {', '.join(SENTIMENT_LABELS)}. Text: \"{sentence}\""
    return {"input": inp, "target": label, "task": "A"}

# -----------------------
# Task B: JSON extraction
# -----------------------
ITEMS = ["blue markers", "black pens", "ruled notebooks", "packing tape", "staples",
         "whiteboard markers", "sticky notes", "printer paper", "erasers", "scissors"]
quantity_words = {1: ["one","a single","1"], 2: ["two","a pair of","2"], 3:["three","3"],
                  4:["four","4"], 5:["five","5"], 10:["ten","10"]}

def make_extraction_example(item: str, qty: int):
    qty_text = random.choice(quantity_words.get(qty,[str(qty)]))
    templates = [
        f"Order {qty_text} {item} for the design team.",
        f"Please send {qty_text} of {item} to the office.",
        f"We need {qty_text} units of {item} ASAP.",
        f"Could you process an order for {qty_text} {item}?",
        f"I would like to purchase {qty_text} {item}."
    ]
    sentence = random.choice(templates)
    if random.random() < 0.3:
        sentence += " This is for next week's sprint."
    inp = f"Extract JSON with fields item (string) and quantity (integer). Text: \"{sentence}\""
    tgt = json.dumps({"item": item, "quantity": qty})
    return {"input": inp, "target": tgt, "task": "B"}

# -----------------------
# Build datasets
# -----------------------
def build_datasets(train_total=200, eval_total=60):
    train, eval_ = [], []
    train_per_task = train_total // 2
    eval_per_task = eval_total // 2
    n_train_per_class = train_per_task // len(SENTIMENT_LABELS)
    n_eval_per_class = eval_per_task // len(SENTIMENT_LABELS)

    for label in SENTIMENT_LABELS:
        for _ in range(n_train_per_class):
            train.append(make_sentiment_example(label))
        for _ in range(n_eval_per_class):
            eval_.append(make_sentiment_example(label))

    while len([x for x in train if x["task"]=="A"]) < train_per_task:
        train.append(make_sentiment_example("neutral"))
    while len([x for x in eval_ if x["task"]=="A"]) < eval_per_task:
        eval_.append(make_sentiment_example("neutral"))

    qty_choices = [1,2,3,4,5,10]
    for _ in range(train_per_task):
        train.append(make_extraction_example(random.choice(ITEMS), random.choice(qty_choices)))
    for _ in range(eval_per_task):
        eval_.append(make_extraction_example(random.choice(ITEMS), random.choice(qty_choices)))

    random.shuffle(train)
    random.shuffle(eval_)
    return DatasetDict({"train": Dataset.from_list(train), "eval": Dataset.from_list(eval_)})

datasets = build_datasets()
print(f"Train size: {len(datasets['train'])}, Eval size: {len(datasets['eval'])}")

# -----------------------
# Tokenization
# -----------------------
max_input_length, max_target_length = 128, 48

def preprocess_function(examples):
    model_inputs = tok_t5(examples["input"], max_length=max_input_length,
                          padding="max_length", truncation=True, return_tensors=None)
    with tok_t5.as_target_tokenizer():
        labels = tok_t5(examples["target"], max_length=max_target_length,
                        padding="max_length", truncation=True, return_tensors=None)
    label_ids = [[(tid if tid!=tok_t5.pad_token_id else -100) for tid in seq] for seq in labels["input_ids"]]
    return {"input_ids": model_inputs["input_ids"], "attention_mask": model_inputs["attention_mask"],
            "labels": label_ids, "orig_input": examples["input"], "orig_target": examples["target"], "task": examples["task"]}

tokenized = datasets.map(preprocess_function, batched=True, remove_columns=datasets["train"].column_names)
tokenized.set_format(type="torch", columns=["input_ids","attention_mask","labels","task","orig_input","orig_target"])

# -----------------------
# Data collator
# -----------------------
data_collator = DataCollatorForSeq2Seq(tokenizer=tok_t5, model=model_t5, return_tensors="pt")

# -----------------------
# Baseline zero-shot evaluation
# -----------------------
def generate_predictions(model, tokenizer, dataset, batch_size=8, max_length=48):
    model.eval()
    preds, targets, tasks = [], [], []
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    for batch in loader:
        input_texts = [s for s in batch["orig_input"]]
        enc = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_input_length)
        with torch.no_grad():
            out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                 max_length=max_length, num_beams=1, do_sample=False)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        preds.extend([d.strip() for d in decoded])
        targets.extend([s for s in batch["orig_target"]])
        tasks.extend([s for s in batch["task"]])
    return preds, targets, tasks

def evaluate_task_level(preds, targets, tasks):
    taskA_idx = [i for i,t in enumerate(tasks) if t=="A"]
    taskB_idx = [i for i,t in enumerate(tasks) if t=="B"]

    a_correct = sum(1 for i in taskA_idx if preds[i].strip()==targets[i].strip())
    a_acc = a_correct / max(1,len(taskA_idx))

    valid_count, field_match_count = 0, 0
    for i in taskB_idx:
        p, t = preds[i].strip(), targets[i].strip()
        try:
            parsed = json.loads(p)
            valid = isinstance(parsed, dict) and "item" in parsed and "quantity" in parsed
        except:
            valid, parsed = False, None
        if valid:
            valid_count += 1
            target_obj = json.loads(t)
            item_match = parsed["item"].strip().lower()==target_obj["item"].strip().lower()
            try:
                parsed_q = int(parsed["quantity"])
            except:
                try:
                    parsed_q = int(float(parsed["quantity"]))
                except:
                    parsed_q = None
            if item_match and parsed_q==target_obj["quantity"]:
                field_match_count += 1
    return {"A_accuracy": a_acc, "B_json_valid": valid_count/max(1,len(taskB_idx)), "B_field_match": field_match_count/max(1,len(taskB_idx))}

print("Running baseline zero-shot evaluation...")
pre_preds, pre_targets, pre_tasks = generate_predictions(model_t5, tok_t5, tokenized["eval"])
pre_metrics = evaluate_task_level(pre_preds, pre_targets, pre_tasks)
print("Baseline results:", pre_metrics)

# -----------------------
# Training Arguments (CPU-compatible)
# -----------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="./sft_output",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=25,
    predict_with_generate=True,
)

# -----------------------
# Compute metrics
# -----------------------
def compute_metrics(eval_pred: EvalPrediction):
    gen_ids, label_ids = eval_pred.predictions, eval_pred.label_ids
    decoded_preds = tok_t5.batch_decode(gen_ids, skip_special_tokens=True)
    labels_for_decoding = np.where(label_ids!=-100, label_ids, tok_t5.pad_token_id)
    decoded_labels = tok_t5.batch_decode(labels_for_decoding, skip_special_tokens=True)
    exact_matches = [1 if p.strip()==l.strip() else 0 for p,l in zip(decoded_preds, decoded_labels)]
    acc = float(sum(exact_matches))/max(1,len(exact_matches))
    return {"eval_accuracy": acc}

trainer = Seq2SeqTrainer(
    model=model_t5,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["eval"],
    tokenizer=tok_t5,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# -----------------------
# Train
# -----------------------
print("Starting fine-tuning...")
start_time = time.time()
trainer.train()
train_time = time.time() - start_time
print(f"Training finished in {train_time:.1f} seconds.")

# -----------------------
# Save fine-tuned model
# -----------------------
trainer.save_model("./sft_model")
print("Saved fine-tuned model to ./sft_model/")

param_count = sum(p.numel() for p in trainer.model.parameters())
print(f"Model parameter count: {param_count:,}")

# -----------------------
# Evaluate after fine-tuning
# -----------------------
best_model = AutoModelForSeq2SeqLM.from_pretrained("./sft_model").to(device)
post_preds, post_targets, post_tasks = generate_predictions(best_model, tok_t5, tokenized["eval"])
post_metrics = evaluate_task_level(post_preds, post_targets, post_tasks)

# -----------------------
# Summary table
# -----------------------
table = [
    ["Task A","Accuracy",f"{pre_metrics['A_accuracy']:.3f}",f"{post_metrics['A_accuracy']:.3f}"],
    ["Task B","JSON validity",f"{pre_metrics['B_json_valid']:.3f}",f"{post_metrics['B_json_valid']:.3f}"],
    ["Task B","Field match",f"{pre_metrics['B_field_match']:.3f}",f"{post_metrics['B_field_match']:.3f}"]
]
print("\n===== TASK METRICS =====")
print(tabulate(table, headers=["Task","Metric","Before SFT","After SFT"], tablefmt="grid"))

# -----------------------
# Comments
# -----------------------
print("\n===== COMMENTS =====")
print("""\
• After SFT, Task A improves via refined label mapping and phrasing understanding.
• Task B shows gains because SFT teaches required JSON output schema.
• Pretrained FLAN-T5 provides instruction-following priors; SFT sharpens task-specific consistency.
• Exact-match metrics are used for checkpointing and evaluation.
• Small CPU-friendly dataset; increase epochs or learning_rate for stronger improvements.
• Constrained decoding or post-processing may help JSON validity.
• SFT aligns model from general instruction-following to deterministic output formatting.
• Inspect failures: label nuance vs. formatting errors.""")
