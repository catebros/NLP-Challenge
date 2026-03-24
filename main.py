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
    "05_training_utils_and_demos.py",
]

for path in SCRIPTS:
    with open(path, "r", encoding="utf-8") as f:
        exec(f.read(), globals())
