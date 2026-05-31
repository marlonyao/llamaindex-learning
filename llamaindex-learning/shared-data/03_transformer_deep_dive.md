Transformer 是一种深度学习架构，由 Vaswani 等人在 2017 年的论文《Attention Is All You Need》中提出。它彻底改变了自然语言处理领域，成为现代大语言模型的基础。

## 核心机制

Transformer 的核心是注意力机制（Attention Mechanism）。与 RNN 和 LSTM 不同，Transformer 不需要按顺序处理数据，可以并行计算整个序列的注意力权重。

### 自注意力（Self-Attention）

自注意力允许模型在编码每个词时，关注输入序列中的所有其他词。计算过程分为三步：

1. 生成 Query、Key、Value 三个向量（通过线性变换）
2. 计算注意力分数：Q × K^T / sqrt(d_k)
3. 加权求和：softmax(分数) × V

### 多头注意力（Multi-Head Attention）

使用多组 Q/K/V 投影，让模型在不同表示子空间中学习不同的注意力模式。通常使用 8 或 16 个头。

## 架构组成

### 编码器（Encoder）

由 N 个相同的层堆叠而成。每层包含：
- 多头自注意力子层
- 前馈神经网络子层
- 层归一化和残差连接

### 解码器（Decoder）

同样由 N 层组成，但每层额外包含：
- 掩码多头自注意力（防止看到未来信息）
- 编码器-解码器注意力

## 位置编码

由于 Transformer 没有循环结构，需要显式注入位置信息。原始论文使用正弦/余弦函数：

PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

现代模型（如 GPT、BERT）通常使用可学习的位置嵌入。

## 对 LLM 的影响

Transformer 架构直接催生了：
- GPT 系列（仅使用解码器）
- BERT（仅使用编码器）
- T5（编码器-解码器完整结构）
- 视觉 Transformer（ViT）

注意力机制的可扩展性使得模型参数量从百万级增长到万亿级，成为"大"语言模型的技术基础。