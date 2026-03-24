"""
Transformer Architectures: BERT -> GPT -> BART
===============================================
Single-file driver that imports all fragments and runs all demos.
"""

import math
import random
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPTS = [
    "00_setup.py",
    "01_batching.py",
    "02_core_modules.py",
    "03_models_bert_bart.py",
    "04_model_gpt_skeleton.py",
]

for path in SCRIPTS:
    with open(path, "r", encoding="utf-8") as f:
        exec(f.read(), globals())


@torch.no_grad()
def estimate_bert_loss(model, eval_iters=20):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses, accs = [], []
        for _ in range(eval_iters):
            x, y = get_classification_batch(split)
            logits, loss = model(x, y)
            preds = logits.argmax(dim=-1)
            losses.append(loss.item())
            accs.append((preds == y).float().mean().item())
        out[split] = {"loss": sum(losses) / len(losses), "acc": sum(accs) / len(accs)}
    model.train()
    return out

@torch.no_grad()
def estimate_bart_loss(model, eval_iters=20):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            src, tgt_in, tgt_out = get_seq2seq_batch(split)
            _, loss = model(src, tgt_in, tgt_out)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

@torch.no_grad()
def estimate_gpt_loss(model, eval_iters=20):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            x, y = get_lm_batch(split)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


# Instantiate models
bert_model = TinyBERT(vocab_size=vocab_size, d_model=d_model, context_length=context_length, n_layers=n_layers, n_classes=2).to(device)
bart_model = TinyBART(vocab_size=vocab_size, d_model=d_model, context_length=context_length, n_layers=n_layers).to(device)
gpt_model = TinyGPT(vocab_size=vocab_size, d_model=d_model, context_length=context_length, n_layers=n_layers).to(device)

# BERT training loop
bert_optimizer = torch.optim.Adam(bert_model.parameters(), lr=learning_rate)

for step in range(2001):
    if step % 50 == 0:
        stats = estimate_bert_loss(bert_model, eval_iters=10)
        print("Step", step, "train loss:", round(stats['train']['loss'], 4), "train acc:", round(stats['train']['acc'], 4), "val loss:", round(stats['val']['loss'], 4), "val acc:", round(stats['val']['acc'], 4))

    xb, yb = get_classification_batch("train")
    logits, loss = bert_model(xb, yb)
    bert_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    bert_optimizer.step()

# BART training loop
bart_optimizer = torch.optim.Adam(bart_model.parameters(), lr=learning_rate)

for step in range(2001):
    if step % 50 == 0:
        losses = estimate_bart_loss(bart_model, eval_iters=10)
        print("Step", step, "train loss:", round(losses['train'], 4), "val loss:", round(losses['val'], 4))

    src, tgt_in, tgt_out = get_seq2seq_batch("train")
    logits, loss = bart_model(src, tgt_in, tgt_out)
    bart_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    bart_optimizer.step()


# GPT training loop
gpt_optimizer = torch.optim.Adam(gpt_model.parameters(), lr=learning_rate)

for step in range(2001):
    if step % 50 == 0:
        losses = estimate_gpt_loss(gpt_model, eval_iters=10)
        print("Step", step, "train loss:", round(losses['train'], 4), "val loss:", round(losses['val'], 4))

    xb, yb = get_lm_batch("train")
    logits, loss = gpt_model(xb, yb)
    gpt_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gpt_optimizer.step()


seed_text = "Natural "
seed_ids = torch.tensor([encode(seed_text)], dtype=torch.long, device=device)

print("\nGreedy decoding")
greedy_out = gpt_model.generate_greedy(seed_ids.clone(), max_new_tokens=150)
print(decode(greedy_out[0].tolist()))

print("\nTemperature sampling (T=0.5)")
temp_out = gpt_model.generate_temperature(seed_ids.clone(), max_new_tokens=150, temperature=0.5)
print(decode(temp_out[0].tolist()))

print("\nTemperature sampling (T=1.5)")
temp_out = gpt_model.generate_temperature(seed_ids.clone(), max_new_tokens=150, temperature=1.5)
print(decode(temp_out[0].tolist()))

print("\nTop-k sampling (k=5, T=1.0)")
topk_out = gpt_model.generate_top_k(seed_ids.clone(), max_new_tokens=150, temperature=1.0, k=5)
print(decode(topk_out[0].tolist()))
