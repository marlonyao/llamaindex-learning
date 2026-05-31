# 阶段一：LlamaIndex 核心概念

## 目标

跑通 LlamaIndex 的完整 RAG 流程，理解 7 个核心概念：
1. **Reader** — 数据加载
2. **Document / Node** — 文档与切分单元
3. **Index** — 索引构建
4. **Retriever** — 检索器
5. **Query Engine** — 查询引擎（Retriever + Synthesizer）
6. **Agent** — 智能体（阶段三深入）
7. **Workflow** — 工作流（阶段三深入）

## 前置条件

```bash
# 1. 安装依赖
cd ../..
pip install -r requirements.txt

# 2. 配置环境变量
# 复制模板文件，填入你的 API Key
cp .env.example .env
# 编辑 .env 文件
```

`.env` 文件格式：
```
WEELINK_API_KEY=sk-xxx
WEELINK_BASE_URL=https://api.weelinking.com/v1
WEELINK_LLM_MODEL=deepseek-v4-flash
WEELINK_EMBED_MODEL=qwen3-embedding-8b
```

## 运行

```bash
cd stage1-basics
python src/stage1_basics.py
```

## 代码结构

```
stage1-basics/
├── src/
│   └── stage1_basics.py    # 主代码，包含 5 个步骤 + bonus
└── README.md
```

## 学习要点

### 步骤 1：Reader
- `SimpleDirectoryReader` 是最常用的 Reader
- 可以过滤文件类型、递归子目录
- 加载后得到 `Document` 列表，每个 Document 有 `.text` 和 `.metadata`

### 步骤 2：Node 切分
- **SimpleNodeParser**：按固定大小切分，最简单
- **chunk_size**：每个 Node 的 token 数（默认 512）
- **chunk_overlap**：相邻 Node 重叠的 token 数，避免切断上下文
- 切分策略直接影响检索质量，阶段二会深入

### 步骤 3：Index
- `VectorStoreIndex` 是最常用的索引类型
- 使用 Weelink 的 `qwen3-embedding-8b` 做 embedding
- 无需 OpenAI 密钥，全部走 Weelink

### 步骤 4：Retriever
- 从索引中返回 Top-K 最相似的 Node
- 可以调整 `similarity_top_k` 控制返回数量
- 返回结果包含 `score`（相似度分数）

### 步骤 5：Query Engine
- 封装了完整的 RAG 流程
- `response_mode` 决定答案合成方式：
  - `compact`：默认，压缩检索结果到提示词
  - `tree_summarize`：分层总结，适合大量结果
  - `refine`：迭代精炼，逐步改进答案
- 返回的 `response.source_nodes` 包含引用来源

## 思考题

1. 为什么 chunk_overlap 很重要？如果设为 0 会发生什么？
2. 如果 similarity_top_k 从 3 改为 10，答案质量会提升吗？为什么？
3. Query Engine 的 response_mode 切换后，输出有什么差异？

## 下一步

完成阶段一后，进入 [阶段二：检索质量深度优化](../stage2-retrieval/README.md)
