# 阶段二：检索质量深度优化

## 目标

掌握"为什么搜不到"和"为什么答不对"的调试能力。

包含 6 个实验：
1. **Chunking 策略对比** — Simple vs Semantic vs Hierarchical
2. **索引类型对比** — VectorStore vs Summary vs DocumentSummary
3. **混合检索** — Vector + BM25
4. **重排序** — LLM Rerank
5. **响应合成模式** — compact vs tree_summarize vs refine
6. **评估指标** — Faithfulness + Relevancy

## 前置条件

```bash
cd ../..
pip install -r requirements.txt
# .env 已配置好
```

## 运行

```bash
cd stage2-retrieval
python src/stage2_retrieval.py
```

## 实验说明

### 实验 1：Chunking 策略对比

三种切分策略对同一查询的检索结果差异。理解：
- SimpleNodeParser：固定大小，可能出现语义切断
- SemanticSplitter：按语义边界，chunk 数量更少但质量更高
- HierarchicalNodeParser：父子层级，适合嵌套文档

### 实验 2：索引类型对比

不同索引的检索机制：
- VectorStoreIndex：向量相似度
- SummaryIndex：全文列表，遍历所有 Node
- DocumentSummaryIndex：先文档摘要，再检索

### 实验 3：混合检索

向量检索 + BM25 关键词检索的融合。理解：
- 向量检索：语义匹配，但可能漏掉关键词
- BM25：关键词精确匹配，但不懂语义
- 融合：取长补短，通常效果更好

### 实验 4：重排序

初步检索 Top-10 → LLM 重排序为 Top-3。理解：
- 初步检索是"快速召回"
- 重排序是"精确筛选"
- 两步策略比一步更准

### 实验 5：响应合成模式

同一查询，不同合成模式的输出差异：
- compact：直接压缩，最简洁
- tree_summarize：分层总结，适合多文档
- refine：迭代精炼，最长但最详细

### 实验 6：评估指标

用自动化指标评估 RAG 质量：
- Faithfulness：答案是否基于检索内容（防幻觉）
- Relevancy：答案是否回答用户问题

## 思考题

1. 为什么 SemanticSplitter 比 SimpleNodeParser 的 chunk 数少？
2. 混合检索的融合策略除了并集，还可以怎么做？
3. 重排序用 LLM 有什么缺点？（成本？延迟？）
4. 评估指标得分低时，应该先调检索还是调提示词？

## 下一步

完成阶段二后，进入 [阶段三：Agent 化](../stage3-agent/README.md)
