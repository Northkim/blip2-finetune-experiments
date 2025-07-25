#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BLIP-2 (OPT-2.7B) QLoRA fine-tune 小样本(2000)训练 + 基础记录
================================================================
本脚本用于对比实验：BLIP-2 上的 QLoRA（4bit + LoRA）微调。

特点 / 与 LoRA baseline 的差异：
  • 使用 bitsandbytes 4bit 量化 (nf4 + double quant) —— 显著降低显存。
  • 在量化模型上仅对 language_model(OPT) 注入 LoRA 适配器 (即 QLoRA)。
  • 冻结除 language_model.* 以外的所有参数；LM 基座 4bit 冻结，仅训练 LoRA。
  • 记录训练耗时、峰值显存、loss 曲线。
  • 随机抽样 2000 条样本，便于硬件对比 (LoRA / AdaLoRA / QLoRA)。

依赖版本建议：
  transformers >= 4.41
  peft        >= 0.11
  bitsandbytes>= 0.43 (或与你 GPU 匹配的版本)
  huggingface_hub >= 0.23

运行：
  python qlora/QLoRA_train_2000.py
"""

import os
import sys
import math
import time
import random
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig,
    Blip2Processor,
    Blip2ForConditionalGeneration,
)

# ------------------------------------------------------------------
# paths
# ------------------------------------------------------------------
PATCHED_DIR = "/mnt/workspace/models/goldsj/blip2-opt-2.7b-patched"  # slow-tokenizer patched model
DATA_DIR    = "/mnt/workspace/data/hmi_dataset"

# 保证能导入 dataset.py
sys.path.append("/mnt/workspace/prepare")
from dataset import HMIDataset  # noqa: E402

# ------------------------------------------------------------------
# device/dtype
# ------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16  # pixel_values dtype + compute dtype fallback

# QLoRA compute dtype: bfloat16 如果支持, 否则 float16
try:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        BNB_COMPUTE_DTYPE = torch.bfloat16
    else:
        BNB_COMPUTE_DTYPE = torch.float16
except AttributeError:
    BNB_COMPUTE_DTYPE = torch.float16

# ------------------------------------------------------------------
# load processor (常规FP16, 不影响量化)
# ------------------------------------------------------------------
print(f"[load] processor from {PATCHED_DIR}")
processor = Blip2Processor.from_pretrained(PATCHED_DIR, local_files_only=True)
tok = processor.tokenizer

# image token
IMAGE_TOKEN = "<image>" if "<image>" in tok.get_vocab() else tok.convert_ids_to_tokens(
    getattr(processor.tokenizer, "image_token_index", 0)
)

# ------------------------------------------------------------------
# load 4bit model (QLoRA)
# ------------------------------------------------------------------
print("[load] 4bit quantized BLIP-2 model ...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",    # NormalFloat4
    bnb_4bit_compute_dtype=BNB_COMPUTE_DTYPE,
)

model = Blip2ForConditionalGeneration.from_pretrained(
    PATCHED_DIR,
    quantization_config=bnb_config,
    device_map="auto",           # accelerate dispatch
    local_files_only=True,
)

# 准备 k-bit 训练 (input grads, layernorm fp32, 等)
print("[prep] prepare_model_for_kbit_training ...")
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

# ------------------------------------------------------------------
# 注入 LoRA 只到 language_model.* (QLoRA)
# ------------------------------------------------------------------
print("[qlora] applying LoRA adapters to language_model only ...")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"],  # OPT attention proj
    task_type="CAUSAL_LM",
)

lm = model.language_model
lm_lora = get_peft_model(lm, peft_config)
model.language_model = lm_lora

# 冻结非 language_model.* 参数 (vision encoder / qformer / projection 等)
for n, p in model.named_parameters():
    if not n.startswith("language_model."):
        p.requires_grad = False

print("[qlora] trainable parameters (lm only):")
model.language_model.print_trainable_parameters()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"[qlora] global trainable params: {trainable:,} / {total:,} = {trainable/total*100:.4f}%")

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
# collate fn
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
# train config
# ------------------------------------------------------------------
EPOCHS           = 1
BATCH_SIZE       = 4
GRAD_ACCUM_STEPS = 1
LR               = 1e-4
MAX_GRAD_NORM    = 1.0

num_batches = math.ceil(len(train_ds) / BATCH_SIZE)
TOTAL_STEPS  = math.ceil(num_batches * EPOCHS / GRAD_ACCUM_STEPS)
print(f"[config] TOTAL_STEPS={TOTAL_STEPS}")

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
# initial sanity loss
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

        if not torch.isfinite(loss):
            print(f"[warn] non-finite loss at step {step}: {loss.item()} -> skip")
            optim.zero_grad(set_to_none=True)
            continue

        loss.backward()

        if (step % GRAD_ACCUM_STEPS) == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optim.step()
            optim.zero_grad(set_to_none=True)

        loss_log.append(loss.item() * GRAD_ACCUM_STEPS)
        if step % 10 == 0:
            print(f"step {step} | loss {loss.item()*GRAD_ACCUM_STEPS:.4f}")

    epoch_time = time.time() - epoch_start
    print(f"[train] epoch {epoch} done in {epoch_time:.2f}s")

# ------------------------------------------------------------------
# timing/mem
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
plt.title("QLoRA Training Loss Curve (2000 samples)")
plt.grid()
plt.savefig("qlora_training_loss_curve_2000.png")
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
# save (LoRA adapters only; base 4bit weights仍来自PATCHED_DIR)
# ------------------------------------------------------------------
output_dir = "./qlora_output_2000/"
os.makedirs(output_dir, exist_ok=True)
model.language_model.save_pretrained(output_dir)
print(f"[save] QLoRA (LoRA-on-4bit) weights saved to {output_dir}")
