#!/usr/bin/env python3
"""
阶段二：检索质量深度优化

目标：理解"为什么搜不到"和"为什么答不对"

包含实验：
1. 多种 Chunking 策略对比（Simple vs Semantic vs Hierarchical）
2. 不同索引类型对比（VectorStore vs PropertyGraph）
3. 混合检索（Vector + BM25）
4. 重排序（Reranking）
5. 响应合成模式对比（compact vs tree_summarize vs refine）
6. 评估指标（Faithfulness, Relevancy）

运行方式：
    python src/stage2_retrieval.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    SummaryIndex,
    PropertyGraphIndex,
)
from llama_index.core.node_parser import (
    SimpleNodeParser,
    SemanticSplitterNodeParser,
    HierarchicalNodeParser,
)
from llama_index.core.retrievers import VectorIndexRetriever, BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.document_summary import DocumentSummaryIndex
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
import numpy as np


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


def experiment1_chunking_strategies(documents, embed_model):
    """
    实验 1：多种 Chunking 策略对比

    对比三种切分策略的效果：
    - SimpleNodeParser: 固定大小切分
    - SemanticSplitter: 按语义边界切分
    - HierarchicalNodeParser: 层级切分（父子关系）
    """
    print("\n" + "=" * 70)
    print("实验 1：Chunking 策略对比")
    print("=" * 70)

    strategies = {}

    # 策略 A：Simple
    simple_parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)
    nodes_simple = simple_parser.get_nodes_from_documents(documents)
    strategies["Simple"] = nodes_simple
    print(f"\n[SimpleNodeParser]")
    print(f"  Node 数: {len(nodes_simple)}")
    print(f"  平均长度: {np.mean([len(n.text) for n in nodes_simple]):.1f} chars")
    print(f"  示例 Node 1: {nodes_simple[0].text[:100]}...")

    # 策略 B：Semantic（需要 embedding 模型）
    semantic_parser = SemanticSplitterNodeParser.from_defaults(
        embed_model=embed_model,
        breakpoint_percentile_threshold=95,
    )
    nodes_semantic = semantic_parser.get_nodes_from_documents(documents)
    strategies["Semantic"] = nodes_semantic
    print(f"\n[SemanticSplitter]")
    print(f"  Node 数: {len(nodes_semantic)}")
    print(f"  平均长度: {np.mean([len(n.text) for n in nodes_semantic]):.1f} chars")
    print(f"  示例 Node 1: {nodes_semantic[0].text[:100]}...")

    # 策略 C：Hierarchical
    hierarchical_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[2048, 512, 128],
    )
    nodes_hierarchical = hierarchical_parser.get_nodes_from_documents(documents)
    strategies["Hierarchical"] = nodes_hierarchical
    print(f"\n[HierarchicalNodeParser]")
    print(f"  Node 数: {len(nodes_hierarchical)}")
    print(f"  平均长度: {np.mean([len(n.text) for n in nodes_hierarchical]):.1f} chars")
    print(f"  示例 Node 1: {nodes_hierarchical[0].text[:100]}...")

    # 检索对比
    query = "Transformer 注意力机制的计算步骤"
    print(f"\n检索对比（查询: {query}）")
    print("-" * 70)

    for name, nodes in strategies.items():
        index = VectorStoreIndex(nodes, embed_model=embed_model)
        retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
        results = retriever.retrieve(query)
        print(f"\n{name}:")
        for i, r in enumerate(results):
            print(f"  [{i+1}] score={r.score:.4f} | {r.node.text[:80]}...")

    return strategies


def experiment2_index_types(documents, embed_model, llm):
    """
    实验 2：不同索引类型对比

    对比：
    - VectorStoreIndex: 向量检索（最常用）
    - SummaryIndex: 全文列表检索
    - DocumentSummaryIndex: 文档摘要索引
    """
    print("\n" + "=" * 70)
    print("实验 2：索引类型对比")
    print("=" * 70)

    # 统一用 SimpleNodeParser
    parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(documents)

    query = "LlamaIndex 和 LangChain 的区别"
    print(f"\n查询: {query}\n")

    # VectorStoreIndex
    print("[VectorStoreIndex]")
    vector_index = VectorStoreIndex(nodes, embed_model=embed_model)
    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=3)
    vector_results = vector_retriever.retrieve(query)
    for i, r in enumerate(vector_results):
        print(f"  [{i+1}] score={r.score:.4f} | {r.node.text[:80]}...")

    # SummaryIndex
    print("\n[SummaryIndex]")
    summary_index = SummaryIndex(nodes)
    summary_engine = summary_index.as_query_engine(llm=llm, similarity_top_k=3)
    summary_response = summary_engine.query(query)
    print(f"  答案: {summary_response.response[:200]}...")

    # DocumentSummaryIndex
    print("\n[DocumentSummaryIndex]")
    doc_summary_index = DocumentSummaryIndex.from_documents(
        documents, embed_model=embed_model, llm=llm
    )
    doc_summary_engine = doc_summary_index.as_query_engine(llm=llm)
    doc_summary_response = doc_summary_engine.query(query)
    print(f"  答案: {doc_summary_response.response[:200]}...")


def experiment3_hybrid_retrieval(documents, embed_model):
    """
    实验 3：混合检索（Vector + BM25）

    向量检索擅长语义匹配，BM25 擅长关键词精确匹配。
    混合通常效果更好。
    """
    print("\n" + "=" * 70)
    print("实验 3：混合检索（Vector + BM25）")
    print("=" * 70)

    parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(documents)

    # 构建向量索引
    vector_index = VectorStoreIndex(nodes, embed_model=embed_model)
    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)

    # 构建 BM25 检索器
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)

    query = "RAG 检索质量"
    print(f"\n查询: {query}\n")

    # 向量检索结果
    print("[Vector 检索]")
    vector_results = vector_retriever.retrieve(query)
    for i, r in enumerate(vector_results[:3]):
        print(f"  [{i+1}] score={r.score:.4f} | {r.node.text[:80]}...")

    # BM25 检索结果
    print("\n[BM25 检索]")
    bm25_results = bm25_retriever.retrieve(query)
    for i, r in enumerate(bm25_results[:3]):
        print(f"  [{i+1}] score={r.score:.4f} | {r.node.text[:80]}...")

    # 简单融合（取并集，去重）
    print("\n[融合检索 - 并集]")
    all_nodes = {}
    for r in vector_results:
        all_nodes[r.node.id_] = r
    for r in bm25_results:
        if r.node.id_ not in all_nodes:
            all_nodes[r.node.id_] = r

    for i, (node_id, r) in enumerate(list(all_nodes.items())[:5]):
        print(f"  [{i+1}] {r.node.text[:80]}...")


def experiment4_reranking(documents, embed_model, llm):
    """
    实验 4：重排序（Reranking）

    初步检索返回 Top-K，但不一定最相关。
    用 LLM 做重排序可以显著提升质量。
    """
    print("\n" + "=" * 70)
    print("实验 4：重排序（Reranking）")
    print("=" * 70)

    parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(documents)

    index = VectorStoreIndex(nodes, embed_model=embed_model)
    query = "大语言模型的幻觉问题"
    print(f"\n查询: {query}\n")

    # 无重排序
    print("[无重排序 - Top-10 检索]")
    retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
    raw_results = retriever.retrieve(query)
    for i, r in enumerate(raw_results[:5]):
        print(f"  [{i+1}] score={r.score:.4f} | {r.node.text[:80]}...")

    # 有 LLM 重排序
    print("\n[LLM 重排序 - Top-10 重排为 Top-3]")
    reranker = LLMRerank(top_n=3, llm=llm)
    reranked_results = reranker.postprocess_nodes(raw_results, query_str=query)
    for i, r in enumerate(reranked_results):
        print(f"  [{i+1}] {r.node.text[:80]}...")


def experiment5_response_modes(documents, embed_model, llm):
    """
    实验 5：响应合成模式对比

    对比三种 response_mode：
    - compact: 默认，压缩到提示词
    - tree_summarize: 分层总结
    - refine: 迭代精炼
    """
    print("\n" + "=" * 70)
    print("实验 5：响应合成模式对比")
    print("=" * 70)

    parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes, embed_model=embed_model)

    query = "请详细解释 Transformer 的注意力机制"
    print(f"\n查询: {query}\n")

    modes = ["compact", "tree_summarize", "refine"]
    for mode in modes:
        print(f"\n[{mode}]")
        print("-" * 50)
        engine = index.as_query_engine(llm=llm, response_mode=mode, similarity_top_k=5)
        response = engine.query(query)
        print(f"{response.response[:300]}...")


def experiment6_evaluation(documents, embed_model, llm):
    """
    实验 6：评估指标

    使用内置的 Evaluator 评估 RAG 质量：
    - Faithfulness: 答案是否忠实于检索内容
    - Relevancy: 答案是否相关于问题
    """
    print("\n" + "=" * 70)
    print("实验 6：评估指标（Faithfulness + Relevancy）")
    print("=" * 70)

    parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    query_engine = index.as_query_engine(llm=llm, similarity_top_k=3)

    # 评估问题集
    eval_queries = [
        "Transformer 的注意力机制是怎么计算的？",
        "RAG 为什么能解决幻觉问题？",
    ]

    faith_evaluator = FaithfulnessEvaluator(llm=llm)
    relev_evaluator = RelevancyEvaluator(llm=llm)

    for query in eval_queries:
        print(f"\n查询: {query}")
        response = query_engine.query(query)
        print(f"答案: {response.response[:150]}...")

        faith_result = faith_evaluator.evaluate_response(query=query, response=response)
        relev_result = relev_evaluator.evaluate_response(query=query, response=response)

        print(f"  Faithfulness: {faith_result.passing} (score: {faith_result.score:.2f})")
        print(f"  Relevancy: {relev_result.passing} (score: {relev_result.score:.2f})")


def main():
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "LlamaIndex 阶段二：检索质量深度优化" + " " * 10 + "║")
    print("╚" + "═" * 68 + "╝")

    api_key, base_url, llm_model, embed_model_name = setup_api()
    if not api_key:
        print("⚠️  请检查 .env 文件中的 WEELINK_API_KEY")
        return

    llm = create_llm(api_key, base_url, llm_model)
    embed_model = create_embed_model(api_key, base_url, embed_model_name)
    documents = load_documents()

    print(f"\n加载了 {len(documents)} 个文档，准备开始 6 个实验...")

    # 运行 6 个实验
    experiment1_chunking_strategies(documents, embed_model)
    experiment2_index_types(documents, embed_model, llm)
    experiment3_hybrid_retrieval(documents, embed_model)
    experiment4_reranking(documents, embed_model, llm)
    experiment5_response_modes(documents, embed_model, llm)
    experiment6_evaluation(documents, embed_model, llm)

    print("\n" + "=" * 70)
    print("阶段二完成！")
    print("=" * 70)
    print("""
你现在已经掌握了：

  ✅ 多种 Chunking 策略（Simple / Semantic / Hierarchical）
  ✅ 不同索引类型（VectorStore / Summary / DocumentSummary）
  ✅ 混合检索（Vector + BM25）
  ✅ 重排序（LLM Rerank）
  ✅ 响应合成模式（compact / tree_summarize / refine）
  ✅ 评估指标（Faithfulness + Relevancy）

接下来进入阶段三：Agent 化。
""")


if __name__ == "__main__":
    main()
