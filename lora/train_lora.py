#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BLIP-2 (OPT-2.7B) LoRA fine-tune minimal training loop **with basic training metrics logging**.

Compared to your original script, this version adds:
  • Config block (EPOCHS, BATCH_SIZE, GRAD_ACCUM_STEPS, LR).
  • Wall-clock timing (total + per-epoch).
  • GPU peak memory capture (if CUDA available).
  • Loss logging list + matplotlib loss curve saved to PNG.
  • Summary printout of key training hyperparams + runtime stats.

Dataset is sub-sampled to at most N_SMALL=2000 examples (seed=42), per your request.

I kept the training logic the same (single forward/backward per batch; optional grad accumulation).
Feel free to expand if you later want validation, checkpoints, JSON logging, WandB, etc.
"""

import os
import random
import time

import torch
import matplotlib.pyplot as plt
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
DTYPE = torch.float16  # keep fp16 to save mem (assumes GPU; may fall back to CPU)


# ------------------------------------------------------------------
# load processor + model
# ------------------------------------------------------------------
print("[load] processor + model from", PATCHED_DIR)
processor = Blip2Processor.from_pretrained(PATCHED_DIR, local_files_only=True)
model = Blip2ForConditionalGeneration.from_pretrained(
    PATCHED_DIR,
    torch_dtype=DTYPE,
    device_map="auto",   # if single GPU, loads there; HF will shard if multi-GPU
    local_files_only=True,
)

# tokenizer + image token handling -------------------------------------------------
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

# quick param stats ------------------------------------------------
trainable, total = 0, 0
for p in model.parameters():
    num = p.numel()
    total += num
    if p.requires_grad:
        trainable += num
print(f"trainable params: {trainable:,} / {total:,} = {trainable/total*100:.4f}%")


# ------------------------------------------------------------------
# dataset small sample (N_SMALL=2000)
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


# ===============================================================
# CONFIG (logging-friendly)
# ===============================================================
EPOCHS = 1
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 1  # set >1 if you need grad accumulation
LR = 1e-4


# ------------------------------------------------------------------
# dataloader (use configured BATCH_SIZE)
# ------------------------------------------------------------------
train_dl = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    collate_fn=collate,
    pin_memory=True,
)


# ------------------------------------------------------------------
# optimizer
# ------------------------------------------------------------------
optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)


# ------------------------------------------------------------------
# 训练 loop + 记录功能
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
        # 训练 step 计数 (global step)
        step = epoch * len(train_dl) + i + 1

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
        loss = out.loss / GRAD_ACCUM_STEPS
        loss.backward()

        if (step % GRAD_ACCUM_STEPS == 0):
            optim.step()
            optim.zero_grad(set_to_none=True)

        # 反标准化记录 (记录真实 loss 数值)
        loss_log.append(loss.item() * GRAD_ACCUM_STEPS)
        if step % 10 == 0:
            print(f"step {step} | loss {loss.item()*GRAD_ACCUM_STEPS:.4f}")

    epoch_time = time.time() - epoch_start
    print(f"[train] epoch {epoch} done in {epoch_time:.2f}s")

# ------------------------------------------------------------------
# Post-train stats
# ------------------------------------------------------------------
total_time = time.time() - start_time
if torch.cuda.is_available():
    max_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
else:
    max_mem = 0.0

print(f"[train] finished. Total time: {total_time:.2f}s")
print(f"[gpu] peak memory usage: {max_mem:.2f} GB")


# ------------------------------------------------------------------
# loss 曲线可视化保存
# ------------------------------------------------------------------
plt.plot(loss_log)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid()
plt.savefig("training_loss_curve.png")
plt.show()


# ------------------------------------------------------------------
# Summary 打印
# ------------------------------------------------------------------
print("==== Summary ====")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Grad accum steps: {GRAD_ACCUM_STEPS}")
print(f"Learning rate: {LR}")
print(f"Total time: {total_time:.2f}s")
print(f"Peak memory: {max_mem:.2f} GB")
print("=================")

# save output
output_dir = "./lora_output_2000//"
os.makedirs(output_dir, exist_ok=True)
model.language_model.save_pretrained(output_dir)
print(f"[save] LoRA adapter weights saved to {output_dir}")