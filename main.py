# ============================================================
# main.py
# Main driver: imports all required modules, then loads and
# executes the course fragments in a shared namespace.
# ============================================================

import math
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

FRAGMENTS = [
    "00_setup.py",
    "01_batching.py",
    "02_core_modules.py",
    "03_models_bert_bart.py",
    "04_model_gpt_skeleton.py",
    "05_training_utils_and_demos.py",
]

for path in FRAGMENTS:
    with open(path, "r", encoding="utf-8") as f:
        exec(f.read(), globals())
