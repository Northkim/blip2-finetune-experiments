#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BLIP-2 (OPT-2.7B) AdaLoRA fine-tune 小样本(2000)训练 + 训练记录 (修复v2)
只训练 language_model 上的 AdaLoRA；冻结 vision / qformer 等非LM模块。
"""

import os
import sys
import math
import time
import random
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from peft import AdaLoraConfig, get_peft_model
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# ------------------------------------------------------------------
# paths
# ------------------------------------------------------------------
PATCHED_DIR = "/mnt/workspace/models/goldsj/blip2-opt-2.7b-patched"
DATA_DIR    = "/mnt/workspace/data/hmi_dataset"

# import dataset
sys.path.append("/mnt/workspace/prepare")
from dataset import HMIDataset  # noqa: E402


# ------------------------------------------------------------------
# device / dtype
# ------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16


# ------------------------------------------------------------------
# load processor + model
# ------------------------------------------------------------------
print(f"[load] processor + model from {PATCHED_DIR}")
processor = Blip2Processor.from_pretrained(PATCHED_DIR, local_files_only=True)
model = Blip2ForConditionalGeneration.from_pretrained(
    PATCHED_DIR,
    torch_dtype=DTYPE,
    device_map="auto",
    local_files_only=True,
)

tok = processor.tokenizer
IMAGE_TOKEN = "<image>" if "<image>" in tok.get_vocab() else tok.convert_ids_to_tokens(
    getattr(model.config, "image_token_index", 0)
)


# ------------------------------------------------------------------
# dataset: random 2k sample
# ------------------------------------------------------------------
print("[data] loading dataset and sampling 2000 examples ...")
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
        txt = IMAGE_TOKEN + " " + cap
        return {"image": img, "text": txt}

train_ds = MiniDataset(_base, idxs)


# ------------------------------------------------------------------
# collate
# ------------------------------------------------------------------
def collate(batch, proc=processor, tok=tok, max_length=64):
    images = [b["image"] for b in batch]
    texts  = [b["text"]  for b in batch]
    enc = proc(
        images=images,
        text=texts,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_length,
    )
    labels = enc["input_ids"].clone()
    labels[labels == tok.pad_token_id] = -100
    enc["labels"] = labels
    return enc


# ------------------------------------------------------------------
# training config
# ------------------------------------------------------------------
EPOCHS           = 1
BATCH_SIZE       = 4
GRAD_ACCUM_STEPS = 1
LR               = 1e-4
MAX_GRAD_NORM    = 1.0   # gradient clipping

num_batches_per_epoch = math.ceil(len(train_ds) / BATCH_SIZE)
TOTAL_STEPS = math.ceil((num_batches_per_epoch * EPOCHS) / GRAD_ACCUM_STEPS)
TINIT  = max(1, int(0.1 * TOTAL_STEPS))
TFINAL = max(TINIT + 1, int(0.8 * TOTAL_STEPS))
print(f"[config] TOTAL_TRAIN_STEPS={TOTAL_STEPS} tinit={TINIT} tfinal={TFINAL}")


# ------------------------------------------------------------------
# AdaLoRA config + patch
# ------------------------------------------------------------------
peft_config = AdaLoraConfig(
    init_r=12,
    target_r=8,
    beta1=0.85,
    beta2=0.85,
    tinit=TINIT,
    tfinal=TFINAL,
    deltaT=10,
    total_step=TOTAL_STEPS,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

lm = model.language_model
lm_adalora = get_peft_model(lm, peft_config)
model.language_model = lm_adalora


# ------------------------------------------------------------------
# freeze everything outside language_model.*  (关键修复)
# ------------------------------------------------------------------
for n, p in model.named_parameters():
    if not n.startswith("language_model."):
        p.requires_grad = False

# PEFT 已在 language_model 内部冻结 base LM 只训练 AdaLoRA 参数
# 打印训练参数数量
print("[adalora] trainable parameters (lm only):")
model.language_model.print_trainable_parameters()

# 全模型层面统计
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"[adalora] global trainable params: {trainable:,} / {total:,} = {trainable/total*100:.4f}%")


# ------------------------------------------------------------------
# dataloader + optim
# ------------------------------------------------------------------
train_dl = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    collate_fn=collate,
    pin_memory=True,
)

optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)


# ------------------------------------------------------------------
# initial sanity loss (before training)
# ------------------------------------------------------------------
print("[debug] computing initial loss on first batch ...")
with torch.no_grad():
    _b = next(iter(train_dl))
    pixel_values   = _b["pixel_values"].to(DEVICE, dtype=DTYPE)
    input_ids      = _b["input_ids"].to(DEVICE)
    attention_mask = _b["attention_mask"].to(DEVICE)
    labels         = _b["labels"].to(DEVICE)
    out0 = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
print(f"[debug] initial loss: {out0.loss.item():.4f}")


# ------------------------------------------------------------------
# training loop
# ------------------------------------------------------------------
loss_log = []
start_time = time.time()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

model.train()
for epoch in range(EPOCHS):
    epoch_start = time.time()
    print(f"[train] epoch {epoch}")
    for i, batch in enumerate(train_dl):
        step = epoch * len(train_dl) + i + 1

        pixel_values   = batch["pixel_values"].to(DEVICE, dtype=DTYPE)
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        out = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = out.loss / GRAD_ACCUM_STEPS

        # NaN / inf guard
        if not torch.isfinite(loss):
            print(f"[warn] non-finite loss at step {step}: {loss.item()}. Skipping update.")
            optim.zero_grad(set_to_none=True)
            continue

        loss.backward()

        if (step % GRAD_ACCUM_STEPS) == 0:
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optim.step()
            optim.zero_grad(set_to_none=True)

        loss_log.append(loss.item() * GRAD_ACCUM_STEPS)

        if step % 10 == 0:
            print(f"step {step} | loss {loss.item()*GRAD_ACCUM_STEPS:.4f}")

    epoch_time = time.time() - epoch_start
    print(f"[train] epoch {epoch} done in {epoch_time:.2f}s")


# ------------------------------------------------------------------
# timing/memory
# ------------------------------------------------------------------
total_time = time.time() - start_time
max_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0

print(f"[train] finished. Total time: {total_time:.2f}s")
print(f"[gpu] peak memory usage: {max_mem:.2f} GB")


# ------------------------------------------------------------------
# plot loss
# ------------------------------------------------------------------
plt.plot(loss_log)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("AdaLoRA Training Loss Curve (2000 samples)")
plt.grid()
plt.savefig("adalora_training_loss_curve_2000.png")
plt.show()


# ------------------------------------------------------------------
# summary
# ------------------------------------------------------------------
print("==== Summary ====")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Grad accum steps: {GRAD_ACCUM_STEPS}")
print(f"Learning rate: {LR}")
print(f"Total time: {total_time:.2f}s")
print(f"Peak memory: {max_mem:.2f} GB")
print("=================")


# ------------------------------------------------------------------
# save
# ------------------------------------------------------------------
output_dir = "./adalora_output_2000/"
os.makedirs(output_dir, exist_ok=True)
model.language_model.save_pretrained(output_dir)
print(f"[save] AdaLoRA weights saved to {output_dir}")
