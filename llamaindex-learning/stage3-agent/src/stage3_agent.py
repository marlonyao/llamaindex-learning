#!/usr/bin/env python3
"""
阶段三：Agent 化

目标：从"单轮问答"升级到"多步推理 + 工具调用"

包含实验：
1. FunctionAgent：Query Engine 作为 Tool
2. 多源 RAG：多个 Query Engine 路由
3. Workflows：事件驱动编排
4. 多步推理：先检索 → 再提取 → 再生成

运行方式：
    python src/stage3_agent.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv(Path(__file__).parent.parent.parent / ".env")

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import FunctionAgent, ReActAgent
from llama_index.core.workflow import (
    Workflow,
    Event,
    StartEvent,
    StopEvent,
    step,
    Context,
)
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding


def setup_api():
    api_key = os.environ.get("WEELINK_API_KEY")
    base_url = os.environ.get("WEELINK_BASE_URL")
    llm_model = os.environ.get("WEELINK_LLM_MODEL", "deepseek-v4-flash")
    embed_model = os.environ.get("WEELINK_EMBED_MODEL", "qwen3-embedding-8b")
    return api_key, base_url, llm_model, embed_model


def create_llm(api_key, base_url, model):
    return OpenAILike(
        model=model,
        api_base=base_url,
        api_key=api_key,
        is_chat_model=True,
        is_function_calling_model=True,
        temperature=0.0,
        max_tokens=2048,
    )


def create_embed_model(api_key, base_url, model):
    return OpenAIEmbedding(
        model=model,
        api_base=base_url,
        api_key=api_key,
        embed_batch_size=10,
    )


def load_documents():
    data_dir = Path(__file__).parent.parent.parent / "shared-data"
    reader = SimpleDirectoryReader(input_dir=str(data_dir), required_exts=[".md"], recursive=True)
    return reader.load_data()


def experiment1_function_agent(documents, embed_model, llm):
    """
    实验 1：FunctionAgent — Query Engine 作为 Tool

    把 RAG 查询引擎包装成 Tool，让 Agent 决定何时调用。
    """
    print("\n" + "=" * 70)
    print("实验 1：FunctionAgent — Query Engine 作为 Tool")
    print("=" * 70)

    parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    query_engine = index.as_query_engine(llm=llm, similarity_top_k=3)

    # 将 Query Engine 包装为 Tool
    rag_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="knowledge_base",
            description="用于查询关于 AI、Transformer、RAG、LlamaIndex 的技术文档知识库",
        ),
    )

    # 创建 FunctionAgent
    agent = FunctionAgent.from_tools(
        tools=[rag_tool],
        llm=llm,
        verbose=True,
    )

    queries = [
        "什么是 Transformer 的注意力机制？",
        "RAG 和微调有什么区别？",
    ]

    for query in queries:
        print(f"\n查询: {query}")
        print("-" * 50)
        response = agent.chat(query)
        print(f"答案: {response.response}")


def experiment2_multi_source_rag(documents, embed_model, llm):
    """
    实验 2：多源 RAG — 多个 Query Engine 路由

    按主题把文档分成不同的索引，Agent 自动路由到正确的索引。
    """
    print("\n" + "=" * 70)
    print("实验 2：多源 RAG — 多个 Query Engine 路由")
    print("=" * 70)

    # 按文档主题分组
    ai_docs = [d for d in documents if "ai" in d.metadata.get("file_name", "")]
    llama_docs = [d for d in documents if "llamaindex" in d.metadata.get("file_name", "")]
    transformer_docs = [d for d in documents if "transformer" in d.metadata.get("file_name", "")]
    rag_docs = [d for d in documents if "rag" in d.metadata.get("file_name", "")]

    parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)

    def build_tool(docs, name, description):
        nodes = parser.get_nodes_from_documents(docs)
        index = VectorStoreIndex(nodes, embed_model=embed_model)
        engine = index.as_query_engine(llm=llm, similarity_top_k=3)
        return QueryEngineTool(
            query_engine=engine,
            metadata=ToolMetadata(name=name, description=description),
        )

    tools = [
        build_tool(ai_docs, "ai_overview", "AI 技术概述、机器学习、深度学习、大模型基础知识"),
        build_tool(llama_docs, "llamaindex_guide", "LlamaIndex 框架使用指南、RAG 构建方法"),
        build_tool(transformer_docs, "transformer_deep", "Transformer 架构详解、注意力机制"),
        build_tool(rag_docs, "rag_techniques", "RAG 技术、检索增强生成、索引策略"),
    ]

    agent = FunctionAgent.from_tools(tools=tools, llm=llm, verbose=True)

    queries = [
        "Transformer 注意力机制的计算步骤是什么？",
        "怎么用 LlamaIndex 构建 RAG 系统？",
    ]

    for query in queries:
        print(f"\n查询: {query}")
        print("-" * 50)
        response = agent.chat(query)
        print(f"答案: {response.response}")


# 实验 3：Workflow 事件定义
class RetrieveEvent(Event):
    query: str
    context: str = ""


class SynthesizeEvent(Event):
    query: str
    retrieved_context: str


class ValidationEvent(Event):
    answer: str
    sources: List[str]


class ResearchWorkflow(Workflow):
    """
    实验 3：Workflow — 事件驱动编排

    三步工作流：
    1. 检索相关文档
    2. 基于检索内容生成答案
    3. 验证答案是否完整
    """

    def __init__(self, index, llm, **kwargs):
        super().__init__(**kwargs)
        self.index = index
        self.llm = llm

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrieveEvent:
        print("\n[Workflow Step 1] 检索相关文档...")
        query = ev.get("query", "")

        query_engine = self.index.as_query_engine(llm=self.llm, similarity_top_k=3)
        response = query_engine.query(query)

        context = "\n".join([n.node.text for n in response.source_nodes])
        print(f"  检索到 {len(response.source_nodes)} 个相关段落")

        return RetrieveEvent(query=query, context=context)

    @step
    async def synthesize(self, ctx: Context, ev: RetrieveEvent) -> SynthesizeEvent:
        print("\n[Workflow Step 2] 合成答案...")

        prompt = f"""基于以下检索内容回答问题：

问题：{ev.query}

检索内容：
{ev.context}

请给出详细、准确的回答。"""

        response = self.llm.complete(prompt)
        return SynthesizeEvent(
            query=ev.query,
            retrieved_context=ev.context,
        )

    @step
    async def validate(self, ctx: Context, ev: SynthesizeEvent) -> StopEvent:
        print("\n[Workflow Step 3] 验证答案完整性...")

        # 简单验证：检查答案是否引用了检索内容
        validation_prompt = f"""判断以下回答是否完整回答了问题：

问题：{ev.query}

检索内容摘要：{ev.retrieved_context[:200]}...

如果回答完整，返回"PASS"。
如果回答不完整，返回"NEED_MORE"并说明原因。
"""

        validation = self.llm.complete(validation_prompt)
        print(f"  验证结果: {validation.text[:100]}")

        return StopEvent(result={
            "query": ev.query,
            "context": ev.retrieved_context,
            "validation": validation.text,
        })


def experiment3_workflow(documents, embed_model, llm):
    """
    实验 3：Workflow — 事件驱动编排
    """
    print("\n" + "=" * 70)
    print("实验 3：Workflow — 事件驱动编排")
    print("=" * 70)

    parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes, embed_model=embed_model)

    import asyncio

    workflow = ResearchWorkflow(index=index, llm=llm, timeout=60)

    query = "解释 Transformer 的多头注意力机制"
    print(f"\n查询: {query}")
    print("-" * 50)

    result = asyncio.run(workflow.run(query=query))
    print(f"\n工作流结果: {result}")


# 实验 4：Workflow 多步推理
class ExtractEvent(Event):
    raw_answer: str


def experiment4_multi_step_reasoning(documents, embed_model, llm):
    """
    实验 4：多步推理 Workflow

    先检索 → 提取结构化数据 → 生成最终报告
    """
    print("\n" + "=" * 70)
    print("实验 4：多步推理 — 先检索 → 再提取 → 再生成")
    print("=" * 70)

    parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    query_engine = index.as_query_engine(llm=llm, similarity_top_k=3)

    query = "对比 LlamaIndex 和 LangChain 的优缺点"
    print(f"\n查询: {query}")
    print("-" * 50)

    # 步骤 1：检索
    print("\n[Step 1] 检索...")
    response = query_engine.query(query)
    context = "\n".join([n.node.text for n in response.source_nodes])
    print(f"  检索到 {len(response.source_nodes)} 段相关内容")

    # 步骤 2：提取结构化数据
    print("\n[Step 2] 提取结构化对比...")
    extract_prompt = f"""从以下内容中提取 LlamaIndex 和 LangChain 的对比点，按以下格式输出：

- LlamaIndex 优势：
- LlamaIndex 劣势：
- LangChain 优势：
- LangChain 劣势：
- 适用场景对比：

内容：
{context}
"""
    extracted = llm.complete(extract_prompt)
    print(f"  提取结果:\n{extracted.text}")

    # 步骤 3：生成最终报告
    print("\n[Step 3] 生成最终报告...")
    report_prompt = f"""基于以下提取的对比点，生成一份简洁的技术选型报告：

{extracted.text}

要求：
1. 结论先行
2. 给出明确选型建议
3. 说明适用场景
"""
    report = llm.complete(report_prompt)
    print(f"  最终报告:\n{report.text}")


def main():
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "LlamaIndex 阶段三：Agent 化" + " " * 15 + "║")
    print("╚" + "═" * 68 + "╝")

    api_key, base_url, llm_model, embed_model_name = setup_api()
    if not api_key:
        print("⚠️  请检查 .env 文件中的 WEELINK_API_KEY")
        return

    llm = create_llm(api_key, base_url, llm_model)
    embed_model = create_embed_model(api_key, base_url, embed_model_name)
    documents = load_documents()

    print(f"\n加载了 {len(documents)} 个文档，准备开始 4 个实验...")

    experiment1_function_agent(documents, embed_model, llm)
    experiment2_multi_source_rag(documents, embed_model, llm)
    experiment3_workflow(documents, embed_model, llm)
    experiment4_multi_step_reasoning(documents, embed_model, llm)

    print("\n" + "=" * 70)
    print("阶段三完成！")
    print("=" * 70)
    print("""
你现在已经掌握了：

  ✅ FunctionAgent — 把 Query Engine 作为 Tool 调用
  ✅ 多源 RAG — 多个索引自动路由
  ✅ Workflow — 事件驱动编排
  ✅ 多步推理 — 检索 → 提取 → 生成

接下来进入阶段四：手写简化版 LlamaIndex。
""")


if __name__ == "__main__":
    main()
