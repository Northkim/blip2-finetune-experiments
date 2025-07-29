# LoRA / AdaLoRA / QLoRA 微调对比报告

##  实验背景与目的

本实验旨在对比三种参数高效微调策略（LoRA / AdaLoRA / QLoRA）在 BLIP-2 (OPT-2.7B) 模型上的性能与硬件消耗，验证不同方法在车载图像描述任务上的适用性。

---

##  统一实验配置

| 配置项             | 值            |
| ------------------ | ------------- |
| 模型               | BLIP-2 OPT-2.7B patched |
| 数据集             | 车载 HMI 数据集（2000 样本） |
| Epochs             | 1             |
| Batch size         | 4             |
| Grad accum steps   | 1             |
| Learning rate      | 1e-4          |
| 设备               | 单 GPU        |
| 框架               | PyTorch + PEFT + transformers |

---

##  核心指标对比

| 指标            | LoRA         | AdaLoRA     | QLoRA       |
| --------------- | ------------ | ----------- | ----------- |
| 总训练时间 (s)  | 59.06        | 65.50       | 215.03      |
| GPU 显存峰值 (GB) | 7.34       | 7.34        | 2.98        |
| step 数         | 500          | 500         | 500         |
| 最后 step loss  | 0.8350       | 0.7007      | 0.9814      |
| 可训练参数比例   | 0.09%（约）  | 0.09%（约）  | 0.09%（约）  |
| 权重保存路径     | ./lora_output_2000/ | ./adalora_output_2000/ | ./qlora_output_2000/ |

---

##  Loss 曲线图

<img src="https://raw.githubusercontent.com/Northkim/blip2-finetune-experiments/main/output.png" style="max-width: 100%; height: auto;" />


---

##  分析与对比总结

### 🔹 硬件消耗对比
- **显存**：QLoRA 显著降低显存（仅 2.98 GB，减少约 60%），非常适合资源受限环境。  
- **时间**：QLoRA 最慢（215s，因 4bit compute overhead），LoRA 最快（59s）。AdaLoRA 稍慢于 LoRA（65s）。

### 🔹 Loss 收敛对比
- **LoRA**：loss 快速下降，最终约 0.8350。
- **AdaLoRA**：初期 loss 高，但下降明显，最终达到 0.7007（最低）。
- **QLoRA**：初期 loss 也快速下降，最终约 0.9814，略高。

### 🔹 可训练参数
- 三种策略可训练参数都在 0.09% 左右，符合“参数高效微调”预期。

---

##  结论建议

- LoRA baseline 速度快，显存要求中等，适合常规 GPU 环境。
- AdaLoRA 在 500 steps 时也能下降到最低 loss（需更长步数体现 rank 调度优势），硬件消耗与 LoRA 接近。
- QLoRA 最大亮点是显存优化（2.98 GB），适合 GPU 资源有限场景，但训练时间最长（215s）。

> 📌 备注：本实验以 2000 样本、1 epoch 为目标，仅比较硬件消耗与初步 loss 收敛趋势。  
> 大数据 / 多 epoch 时 AdaLoRA 可能展现动态 rank 优势，QLoRA 的存储与 IO 开销也可能 amortize。

---

 
