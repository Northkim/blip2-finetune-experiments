#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA fine-tune BLIP-2 (OPT-2.7B) minimal manual training loop (v3)
- Uses *patched* model dir with <image> token + config fields.
- Applies LoRA ONLY to the language_model (OPT) submodule to avoid forward kwarg clashes.
- Small random sample (~2000) for dry run.

Run:  python train_lora_manual_v3.py
"""

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from dataset import HMIDataset  # your dataset returning (PIL.Image or np.array, caption_str)

# ------------------------------------------------------------------
# paths
# ------------------------------------------------------------------
PATCHED_DIR = "/mnt/workspace/models/goldsj/blip2-opt-2.7b-patched"  # from patch_blip2.py
DATA_DIR = "/mnt/workspace/data/hmi_dataset"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16  # keep fp16 to save mem

# ------------------------------------------------------------------
# load processor + model
# ------------------------------------------------------------------
print("[load] processor + model from", PATCHED_DIR)
processor = Blip2Processor.from_pretrained(PATCHED_DIR, local_files_only=True)
model = Blip2ForConditionalGeneration.from_pretrained(
    PATCHED_DIR,
    torch_dtype=DTYPE,
    device_map="auto",   # if single GPU, loads there
    local_files_only=True,
)

tok = processor.tokenizer
IMAGE_TOKEN = "<image>" if "<image>" in tok.get_vocab() else tok.convert_ids_to_tokens(getattr(model.config, "image_token_index", 0))

# ------------------------------------------------------------------
# apply LoRA **only** to the decoder LM (model.language_model)
# ------------------------------------------------------------------
print("[lora] wrapping language_model only ...")
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"],  # OPT attn Q/V
    task_type="CAUSAL_LM",  # decoder-only LM
)

lm = model.language_model  # OPTForCausalLM
lm_lora = get_peft_model(lm, peft_config)
model.language_model = lm_lora  # swap in

# (optional) freeze everything outside LoRA (safe but not required; PEFT already handles)
for n, p in model.named_parameters():
    if "lora_A" in n or "lora_B" in n:
        p.requires_grad = True
    else:
        # ensure base weights frozen (cheap fine-tune)
        p.requires_grad = False

# quick param stats
trainable, total = 0, 0
for p in model.parameters():
    num = p.numel()
    total += num
    if p.requires_grad:
        trainable += num
print(f"trainable params: {trainable:,} / {total:,} = {trainable/total*100:.4f}%")

# ------------------------------------------------------------------
# dataset small sample
# ------------------------------------------------------------------
print("[data] loading base dataset ...")
_base = HMIDataset(DATA_DIR)
N_SMALL = 2000
idxs = list(range(len(_base)))
if len(idxs) > N_SMALL:
    idxs = random.Random(42).sample(idxs, N_SMALL)
print(f"[data] sampled {len(idxs)} / {len(_base)}")

class MiniDataset(Dataset):
    def __init__(self, base, indices):
        self.base = base
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        img, cap = self.base[self.indices[i]]
        # prepend <image> token so processor knows where image goes
        txt = IMAGE_TOKEN + " " + cap
        return {"image": img, "text": txt}

train_ds = MiniDataset(_base, idxs)

# ------------------------------------------------------------------
# collate → processor encodes both image + text, builds labels
# ------------------------------------------------------------------

def collate(batch, proc=processor, tok=tok, max_length=64):
    images = [b["image"] for b in batch]
    texts  = [b["text"]  for b in batch]

    enc = proc(
        images=images,
        text=texts,
        return_tensors="pt",
        padding="longest",   # dynamic pad
        truncation=True,
        max_length=max_length,
    )
    # labels: ignore padding
    labels = enc["input_ids"].clone()
    pad_id = tok.pad_token_id
    labels[labels == pad_id] = -100
    enc["labels"] = labels
    return enc

train_dl = DataLoader(
    train_ds,
    batch_size=2,    # small; increase if mem ok
    shuffle=True,
    num_workers=4,
    collate_fn=collate,
    pin_memory=True,
)

# ------------------------------------------------------------------
# optimizer
# ------------------------------------------------------------------
optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# ------------------------------------------------------------------
# train loop (1 epoch dry run)
# ------------------------------------------------------------------
model.train()
EPOCHS = 2
step = 0
for epoch in range(EPOCHS):
    print(f"[train] epoch {epoch}")
    for batch in train_dl:
        step += 1
        # move to device; pixel_values to model dtype
        pixel_values = batch["pixel_values"].to(DEVICE, dtype=DTYPE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        out = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = out.loss
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)

        if step % 10 == 0:
            print(f"step {step} | loss {loss.item():.4f}")



print("[train] finished 50 steps smoke test.")

print("done.")
