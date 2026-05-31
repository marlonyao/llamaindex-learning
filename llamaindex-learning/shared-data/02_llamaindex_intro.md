LlamaIndex 是一个开源的数据框架，专门用于构建基于大语言模型的检索增强生成（RAG）应用。它的核心使命是将 LLM 与企业内部数据连接起来。

## 核心设计哲学

LlamaIndex 的设计围绕"数据优先"的理念。与 LangChain 这样的通用编排框架不同，LlamaIndex 把数据摄取、索引和检索作为一等公民。其架构抽象层直接映射到 RAG 流水线的各个阶段：

1. 读取（Readers）
2. 切分（Documents → Nodes）
3. 索引（Indexes）
4. 检索（Retrievers）
5. 合成（Query Engines）

## 关键概念

### Document 和 Node

Document 是原始内容单元，带有元数据。Node 是 Document 的切片，是索引和检索的实际单位。LlamaIndex 提供多种 NodeParser 来将文档切分为语义上有意义的块：

- SimpleNodeParser：按固定大小切分
- SemanticSplitter：按语义边界切分
- HierarchicalNodeParser：构建父子层级关系
- MarkdownNodeParser：按 Markdown 结构切分

### 索引类型

- VectorStoreIndex：最常用，基于向量相似度检索
- SummaryIndex：全文摘要索引
- TreeIndex：树形层级索引
- PropertyGraphIndex：基于知识图谱的索引（2024 年推荐）

### Query Engine

Query Engine 是检索器（Retriever）和响应合成器（Response Synthesizer）的组合。它封装了完整的 RAG 流程：接收查询 → 检索相关节点 → 合成答案。

响应合成模式包括：
- compact：默认模式，将检索结果压缩到提示词中
- tree_summarize：分层总结，适合大量检索结果
- refine：迭代精炼，逐步改进答案

## Workflows

Workflows 是 LlamaIndex 的事件驱动编排框架。它使用 @step 装饰器定义工作流节点，节点通过类型化事件进行通信。Workflows 支持并行执行、循环和状态管理，适合构建复杂的 Agent 流程。

## 生态工具

- LlamaParse：企业级文档解析，支持复杂表格、图表和手写文字
- LlamaCloud：托管的 RAG 服务
- LlamaExtract：结构化数据提取

## 与 LangChain 的关系

LlamaIndex 和 LangChain 不是竞争关系。生产环境中的常见模式是：LlamaIndex 负责检索层，LangGraph 负责 Agent 编排层。两者通过 LlamaIndex 的 Query Engine 作为 LangChain 的 Tool 来集成。