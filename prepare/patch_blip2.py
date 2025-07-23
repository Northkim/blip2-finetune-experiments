#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
安全补丁：为 BLIP-2 模型添加 <image> token & pad_token，并强制使用慢速 tokenizer。
避免 fast tokenizer JSON 被破坏导致后续加载报错。
"""

import os
import json
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    Blip2ForConditionalGeneration,
    AddedToken,
)

# ----- 路径 -----
SRC_DIR = Path("/mnt/workspace/models/goldsj/blip2-opt-2.7b")          # 下载原始模型
DST_DIR = Path("/mnt/workspace/models/goldsj/blip2-opt-2.7b-patched")  # 补丁后存放
DST_DIR.mkdir(parents=True, exist_ok=True)

print("[patch] loading base model + components from:", SRC_DIR)

# --- tokenizer: 强制慢速 ---
tok = AutoTokenizer.from_pretrained(
    str(SRC_DIR),
    use_fast=False,
    local_files_only=True,
)

# pad_token fix
if tok.pad_token is None:
    tok.pad_token = tok.eos_token  # 或者 tok.add_special_tokens({"pad_token": "<pad>"})

# add <image> special token
if "<image>" not in tok.get_vocab():
    print("[patch] adding <image> token ...")
    tok.add_tokens([AddedToken("<image>", normalized=False, special=True)], special_tokens=True)

# --- load image processor ---
image_proc = AutoImageProcessor.from_pretrained(
    str(SRC_DIR),
    local_files_only=True,
)

# --- load model ---
model = Blip2ForConditionalGeneration.from_pretrained(
    str(SRC_DIR),
    torch_dtype=torch.float16,
    local_files_only=True,
)

# resize embeddings after token add
model.resize_token_embeddings(len(tok))

# set config fields
image_token_id = tok.convert_tokens_to_ids("<image>")
model.config.image_token_index = int(image_token_id)
if getattr(model.config, "num_query_tokens", None) is None:
    # fallback 推断
    if hasattr(model, "qformer") and hasattr(model.qformer, "config"):
        model.config.num_query_tokens = getattr(model.qformer.config, "num_query_tokens", 32)
    else:
        model.config.num_query_tokens = 32

print("[patch] saving patched model →", DST_DIR)

# save model
model.save_pretrained(str(DST_DIR), safe_serialization=True)

# save tokenizer (慢速，避免写 fast tokenizer.json)
tok.save_pretrained(str(DST_DIR))

# save image processor
image_proc.save_pretrained(str(DST_DIR))

# write tokenizer_config 强制 slow（保险）
tok_cfg_path = DST_DIR / "tokenizer_config.json"
if tok_cfg_path.exists():
    # merge with existing
    try:
        existing = json.loads(tok_cfg_path.read_text())
    except Exception:
        existing = {}
else:
    existing = {}
existing.update({"use_fast": False})
tok_cfg_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))

# 如果 transformers 自己生成了 tokenizer.json（fast），删掉它，避免再次触发
fast_tok = DST_DIR / "tokenizer.json"
if fast_tok.exists():
    fast_tok.rename(DST_DIR / "tokenizer.json.bak_do_not_use")

print("[patch] done. patched model at:", DST_DIR)
