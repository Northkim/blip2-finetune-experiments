#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BLIP-2 (OPT-2.7B) Adapter fine-tune 全数据训练 + 基础记录
"""

import os
import random
import time
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from peft import AdaLoraConfig, get_peft_model
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from dataset import HMIDataset

PATCHED_DIR = "/mnt/workspace/models/goldsj/blip2-opt-2.7b-patched"
DATA_DIR = "/mnt/workspace/data/hmi_dataset"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

processor = Blip2Processor.from_pretrained(PATCHED_DIR, local_files_only=True)
model = Blip2ForConditionalGeneration.from_pretrained(
    PATCHED_DIR,
    torch_dtype=DTYPE,
    device_map="auto",
    local_files_only=True,
)

tok = processor.tokenizer
IMAGE_TOKEN = "<image>" if "<image>" in tok.get_vocab() else tok.convert_ids_to_tokens(getattr(model.config, "image_token_index", 0))

# Adapter Config
peft_config = AdaLoraConfig(
    init_r=12, target_r=8, beta1=0.85, beta2=0.85, tinit=200, tfinal=1000, deltaT=10,
    lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM"
)

lm = model.language_model
lm_adapter = get_peft_model(lm, peft_config)
model.language_model = lm_adapter

for n, p in model.named_parameters():
    p.requires_grad = ("lora_A" in n or "lora_B" in n)

trainable, total = 0, 0
for p in model.parameters():
    num = p.numel()
    total += num
    if p.requires_grad:
        trainable += num
print(f"trainable params: {trainable:,} / {total:,} = {trainable/total*100:.4f}%")

print("[data] using full dataset ...")
_base = HMIDataset(DATA_DIR)

class FullDataset(Dataset):
    def __init__(self, base):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        img, cap = self.base[i]
        txt = IMAGE_TOKEN + " " + cap
        return {"image": img, "text": txt}

train_ds = FullDataset(_base)
print(f"[data] dataset size = {len(train_ds)}")

def collate(batch, proc=processor, tok=tok, max_length=64):
    images = [b["image"] for b in batch]
    texts  = [b["text"]  for b in batch]
    enc = proc(images=images, text=texts, return_tensors="pt", padding="longest", truncation=True, max_length=max_length)
    labels = enc["input_ids"].clone()
    labels[labels == tok.pad_token_id] = -100
    enc["labels"] = labels
    return enc

EPOCHS = 1
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 1
LR = 1e-4

train_dl = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
    collate_fn=collate, pin_memory=True,
)

optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

loss_log = []
start_time = time.time()
torch.cuda.reset_peak_memory_stats()

model.train()
for epoch in range(EPOCHS):
    epoch_start = time.time()
    print(f"[train] epoch {epoch}")
    for i, batch in enumerate(train_dl):
        step = epoch * len(train_dl) + i + 1

        pixel_values = batch["pixel_values"].to(DEVICE, dtype=DTYPE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        out = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss / GRAD_ACCUM_STEPS
        loss.backward()

        if (step % GRAD_ACCUM_STEPS == 0):
            optim.step()
            optim.zero_grad(set_to_none=True)

        loss_log.append(loss.item() * GRAD_ACCUM_STEPS)
        if step % 10 == 0:
            print(f"step {step} | loss {loss.item()*GRAD_ACCUM_STEPS:.4f}")

    epoch_time = time.time() - epoch_start
    print(f"[train] epoch {epoch} done in {epoch_time:.2f}s")

total_time = time.time() - start_time
max_mem = torch.cuda.max_memory_allocated() / 1024**3

print(f"[train] finished. Total time: {total_time:.2f}s")
print(f"[gpu] peak memory usage: {max_mem:.2f} GB")

plt.plot(loss_log)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid()
plt.savefig("adapter_training_loss_curve.png")
plt.show()

print("==== Summary ====")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Grad accum steps: {GRAD_ACCUM_STEPS}")
print(f"Learning rate: {LR}")
print(f"Total time: {total_time:.2f}s")
print(f"Peak memory: {max_mem:.2f} GB")
print("=================")

# 保存 Adapter 权重
output_dir = "./adapter_output/"
os.makedirs(output_dir, exist_ok=True)
model.language_model.save_pretrained(output_dir)
print(f"[save] Adapter weights saved to {output_dir}")
