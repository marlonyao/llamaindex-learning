# LlamaIndex 学习项目

分四个阶段，从"能跑通"到"能写框架"。

## 阶段规划

| 阶段 | 主题 | 目标 | 预计时间 |
|---|---|---|---|
| **阶段一** | 核心概念 | 跑通 RAG 全流程，理解 7 个核心概念 | 1-2 天 |
| **阶段二** | 检索质量 | 掌握 chunking、混合检索、重排序 | 3-5 天 |
| **阶段三** | Agent 化 | FunctionAgent + Workflows 编排 | 2-3 天 |
| **阶段四** | 手写框架 | 实现简化版 LlamaIndex | 5-7 天 |

## 共享数据

`shared-data/` 目录包含 4 个 Markdown 文档，用于所有阶段的测试：
- `01_ai_overview.md` — AI 概述
- `02_llamaindex_intro.md` — LlamaIndex 介绍
- `03_transformer_deep_dive.md` — Transformer 详解
- `04_rag_techniques.md` — RAG 技术

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
# 复制 .env.example 为 .env，填入你的 API Key
cp .env.example .env
# 编辑 .env 文件，填入 Weelink API Key

# 3. 从阶段一开始
cd stage1-basics
python src/stage1_basics.py
```

## 环境变量配置

`.env` 文件（已包含 `.gitignore`，不会提交到 Git）：

```
WEELINK_API_KEY=sk-your-key
WEELINK_BASE_URL=https://api.weelinking.com/v1
WEELINK_LLM_MODEL=deepseek-v4-flash
WEELINK_EMBED_MODEL=qwen3-embedding-8b
```

## 目录结构

```
llamaindex-learning/
├── .env.example                  # 环境变量模板
├── .env                          # 真实密钥（不提交 Git）
├── .gitignore                    # 忽略 .env 等文件
├── requirements.txt
├── README.md
├── shared-data/                  # 测试文档
│   ├── 01_ai_overview.md
│   ├── 02_llamaindex_intro.md
│   ├── 03_transformer_deep_dive.md
│   └── 04_rag_techniques.md
├── stage1-basics/                # 阶段一：核心概念
│   ├── src/stage1_basics.py
│   └── README.md
├── stage2-retrieval/             # 阶段二：检索质量（待开发）
│   ├── src/
│   ├── tests/
│   └── README.md
├── stage3-agent/                 # 阶段三：Agent 化（待开发）
│   ├── src/
│   ├── tests/
│   └── README.md
└── stage4-mini-llamaindex/       # 阶段四：手写框架（待开发）
    ├── src/
    ├── tests/
    └── README.md
```

## 学习建议

1. **每个阶段完成后，先理解再推进**。不要跳过思考题。
2. **修改代码参数**，看效果变化。比如改 chunk_size、similarity_top_k、response_mode。
3. **用你自己的文档替换 shared-data**，测试真实场景。

## 参考资源

- [LlamaIndex 官方文档](https://docs.llamaindex.ai/)
- [LlamaIndex Cookbook](https://docs.llamaindex.ai/en/stable/examples/)
- [LlamaIndex Workflows](https://docs.llamaindex.ai/en/stable/module_guides/workflow/)
