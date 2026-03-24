# ACTIVITY

## Transformer Architectures: BERT → GPT → BART

---

## Overview

This activity is designed to help you understand how different transformer architectures emerge from small but critical changes.

You will:

- Work with a **BERT-style encoder**
- Inspect a **BART-style encoder-decoder**
- Build a **GPT-style decoder-only model from scratch**

The goal is not just to implement models, but to answer:

> What architectural change turns one transformer into another?

All explanations should be grounded in outputs from your own models. :contentReference[oaicite:0]{index=0}

---

## Project Structure

The codebase is split into modular fragments:

| File | Purpose |
|------|--------|
| `00_setup.py` | Dataset, vocabulary, train/val split, hyperparameters |
| `01_batching.py` | Batch generation for LM, classification, seq2seq |
| `02_core_modules.py` | Attention + FeedForward + block skeletons |
| `03_models_bert_bart.py` | Full TinyBERT and TinyBART |
| `04_model_gpt_skeleton.py` | GPT scaffold (you complete this) |
| `05_training_utils_and_demos.py` | Training loops (GPT missing) |
| `99_loader_example.py` | Example entry point |

---

## What You Need to Do

---

## Task 0 — Setup

- Add missing imports (`torch`, `nn`, `F`, `math`, etc.)
- Create a main driver file
- Load fragments using `exec(...)`
- Run BERT and BART demos to confirm everything works

---

## Task 1 — Hyperparameters

Fill in:

- `batch_size`
- `context_length`
- `d_model`
- `n_layers`
- `learning_rate`

Start small (fast training).

### Think about:

- Which parameters affect **model capacity** vs **training dynamics**
- Why context length matters differently for:
  - classification (BERT)
  - generation (GPT)
- Tradeoffs of increasing:
  - model size
  - number of layers

---

## Task 2 — Attention

Study `AttentionHead`.

### Experiment:

- Disable causal masking
- Later compare GPT outputs with:
  - masking ON
  - masking OFF

### Goal:

Understand how **masking creates directionality in time**

---

## Task 3 — Complete Blocks

Implement:

- `EncoderBlock` (already done)
- `DecoderBlock`
- `EncoderDecoderBlock`

Use the residual pattern:

```

x = x + attention(...)
x = x + feedforward(...)

```

### Think about:

- Where does encoder information enter in BART?
- Why GPT/BERT don’t need cross-attention?
- What happens if masking is removed?

---

## Task 4 — Build TinyGPT

Complete:

```

04_model_gpt_skeleton.py

```

You must:

- Add token + positional embeddings
- Stack decoder blocks
- Add final layer norm
- Project to vocabulary logits
- Implement next-token loss

### Core idea:

> GPT = decoder-only transformer + language modeling head

---

## Task 5 — Decoding Strategies

Implement:

- Greedy decoding
- Temperature sampling
- Top-k sampling

### Compare outputs:

- temperature = 0.5 vs 1.5
- low vs high randomness
- coherence vs diversity

### Think about:

- Why temperature affects entropy
- Why top-k avoids bad tokens

---

## Task 6 — GPT Training + Demo

In `05_training_utils_and_demos.py`:

- Instantiate TinyGPT
- Train using `get_lm_batch()`
- Evaluate using `estimate_gpt_loss()`
- Generate text using:
  - greedy
  - temperature
  - top-k

---

## Reflection Questions

---

### Architecture

- What makes GPT **decoder-only**?
- Why does GPT use next-token prediction?
- Why can’t BERT generate text properly?

---

### Attention + Masking

- What happens when causal masking is removed?
- Which version behaves autoregressively?
- Why?

---

### Decoding

- Which method is:
  - most coherent?
  - most diverse?
- Does lower loss = better text?

---

### Model Comparison

Run all models and analyze:

- BERT output (classification)
- GPT output (generation)
- BART output (seq2seq)

Explain:

> How masking + attention structure → different behaviors

---

## Key Insight

Everything reduces to this:

- **BERT**
  → bidirectional attention  
  → encoder-only  

- **GPT**
  → causal masking  
  → decoder-only  
  → autoregressive  

- **BART**
  → encoder + decoder  
  → cross-attention  

---

## Deliverables

You should have:

- Working TinyGPT implementation
- Training + generation outputs
- Answers to all questions
- Clear explanation of architectural differences

---

## End Goal

By the end, you should be able to answer:

> How does one transformer architecture turn into another?


And prove it with your own code + outputs.
