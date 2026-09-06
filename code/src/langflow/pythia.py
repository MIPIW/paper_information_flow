"""Loading Pythia. Activation capture and architecture-agnostic accessors
live in `langflow.models`, shared with `langflow.bloom`.
"""

import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

# Re-exported so existing call sites (`from langflow.pythia import ...`) keep working.
from langflow.models import ModelActivations as PythiaActivations  # noqa: F401
from langflow.models import down_proj_weight, run_with_cache, tokenize_batch  # noqa: F401


def load_pythia(model_name: str = "EleutherAI/pythia-70m-deduped", device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Force float32: several checkpoints in this family default to float16, whose
    # ~65504 max magnitude overflows on the large-norm residual-stream dimensions
    # ("massive activations") that show up in the last layer of small models.
    model = GPTNeoXForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.to(device)
    model.eval()
    return model, tokenizer
