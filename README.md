# blip2-finetune-experiments
实习时要求对blip2进行微调策略对比：Adalora/LoRA/QLoRA硬件消耗实测 因为之前没有接触过模型微调，大部分代码在ai指导下生成～ 这个仓库的创建目的是记录一下，水平很低，理解很差😭🥹🥹

### 硬件消耗报告
[点击这里](./report.md)

### 数据以及数据预处理
- 数据获取详情请见modelscope平台，[具体链接：](https://www.modelscope.cn/datasets/Northkim/archive_processed_data_by_class)<br>
- 数据预处理代码可见[01_preprocess.ipynb](./01_preprocess.ipynb)

### lora 微调
[lora微调部分可见:](./lora)

### adalora 微调
[adalora微调部分可见:](./adapter)

### qlora 微调
[qlora微调部分可见:](./qlora)




