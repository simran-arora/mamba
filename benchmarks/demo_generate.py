import argparse
import time
import json

import torch
import torch.nn.functional as F

from einops import rearrange
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.append("/home/bfs/simran/clean4/ThunderKittens/demos/mamba2/mamba/")
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

device = "cuda"
dtype = torch.float16
use_tk = True
genlen = 1
model_name = "state-spaces/mamba2-370m"

print(f"Loading model {model_name}")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
model = MambaLMHeadModel.from_pretrained(model_name, device=device, dtype=dtype, use_tk=use_tk)
model.eval()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

prompt = "The quick brown fox jumps over the lazy"

torch.random.manual_seed(0)
tokens = tokenizer(prompt, return_tensors="pt")
input_ids = tokens.input_ids.to(device=device)
attn_mask = tokens.attention_mask.to(device=device)
max_length = input_ids.shape[1] + genlen

out = model.generate(
    input_ids=input_ids,
    max_length=max_length,
    cg=False,
    return_dict_in_generate=True,
    output_scores=True,
    enable_timing=False,
    temperature=1.0,
    top_k=1.0,
    top_p=1.0,
    min_p=0.0,
    repetition_penalty=1.0,
)

print(f"Use tk: {use_tk}")
print(tokenizer.batch_decode(out.sequences.tolist()))


