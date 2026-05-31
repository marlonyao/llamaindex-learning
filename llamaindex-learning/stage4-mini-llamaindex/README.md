# 阶段四：手写简化版 LlamaIndex

## 目标

理解框架背后的设计原理，而不只是 API 调用。

## 核心设计

**手写版特点**：
- 最小依赖：只用 numpy 和 LLM API 调用
- 展示原理：每个组件都有"手写版"和"官方版"对比
- 可运行：能用共享数据跑通完整 RAG

## 手写版组件

| 组件 | 手写版 | 官方版 | 差异说明 |
|------|--------|--------|----------|
| Document | dataclass | Document 类 | 官方版支持更多元数据 |
| Node | dataclass | Node 类 | 官方版支持关系图、层级 |
| NodeParser | 按字符切分 | 多种策略 | 官方版有 Semantic/Hierarchical |
| Embedding | Mock（字符频率） | API 调用 | 手写版展示原理，真实版用 Weelink |
| VectorStore | 内存 numpy 数组 | 可插拔存储 | 官方版支持 Qdrant/Pinecone |
| 相似度计算 | 暴力 dot product | ANN (HNSW) | 官方版用近似最近邻加速 |
| Query Engine | 简单 prompt 拼接 | 多种合成模式 | 官方版有 compact/tree/refine |

## 运行

```bash
cd stage4-mini-llamaindex
python src/stage4_mini_llamaindex.py
```

## 实验说明

### 实验 1：手写版 vs 官方版对比

展示核心差异表格，理解官方版为什么需要那么多抽象。

### 实验 2：运行手写版完整 RAG

用共享数据跑通：Reader → Parser → Index → Query Engine。

如果配置了 Weelink API Key，会调用真实 API；
如果没有配置，会用 Mock Embedding 演示原理。

### 实验 3：理解数学原理

展示余弦相似度的计算过程，用简单向量演示。

### 实验 4：性能对比

暴力检索 vs 近似最近邻（ANN）的性能差异。

## 手写版核心代码量

```
Document/Node:       ~20 行
NodeParser:          ~30 行
Embedding (Mock):    ~30 行
VectorStoreIndex:    ~40 行
Query Engine:        ~40 行
Reader:              ~20 行
总计:               ~180 行
```

## 思考题

1. 为什么 Node 要记录 start_char_idx 和 end_char_idx？
2. chunk_overlap 在代码中是怎么实现的？如果设为 0 会怎样？
3. Mock Embedding 用字符频率统计，为什么能"工作"？
4. 暴力检索的 O(N) 复杂度在实际中有什么影响？
5. 如果要扩展手写版支持 100 万条数据，需要改哪里？

## 学习建议

完成手写版后，建议阅读官方 LlamaIndex 源码：
- `llama_index/core/schema.py` — Document / Node 定义
- `llama_index/core/node_parser.py` — 切分策略
- `llama_index/core/indices/vector_store/base.py` — VectorStoreIndex
- `llama_index/core/query_engine/` — Query Engine 实现

对比手写版和官方版，理解每个设计决策的 trade-off。

## 恭喜

四个阶段全部完成！你已经从 API 调用者成长为理解底层原理的开发者。
