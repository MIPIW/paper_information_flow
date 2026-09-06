"""Loading BLOOM. Activation capture and architecture-agnostic accessors live
in `langflow.models`, shared with `langflow.pythia`.
"""

import torch
from transformers import AutoTokenizer, BloomForCausalLM


def load_bloom(model_name: str = "bigscience/bloom-1b7", device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = BloomForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.to(device)
    model.eval()
    return model, tokenizer
