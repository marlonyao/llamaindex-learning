#!/usr/bin/env python3
"""
阶段四：手写简化版 LlamaIndex

目标：理解框架背后的设计原理，而不只是 API 调用。

核心设计：
- 最小依赖：只用 numpy 和 LLM API 调用
- 展示原理：每个组件都有"手写版"和"官方版"对比
- 可运行：能用共享数据跑通完整 RAG

运行方式：
    python src/stage4_mini_llamaindex.py
"""

import os
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

import numpy as np

load_dotenv(Path(__file__).parent.parent.parent / ".env")


# ============================================================================
# 手写版：核心抽象
# ============================================================================

@dataclass
class Document:
    """手写版 Document：原始内容单元"""
    text: str
    metadata: Dict = field(default_factory=dict)
    id_: str = field(default_factory=lambda: hashlib.md5(os.urandom(16)).hexdigest()[:12])

    def __repr__(self):
        return f"Document(id={self.id_}, text={self.text[:50]}..., meta={self.metadata})"


@dataclass
class Node:
    """手写版 Node：切分后的检索单元"""
    text: str
    metadata: Dict = field(default_factory=dict)
    id_: str = field(default_factory=lambda: hashlib.md5(os.urandom(16)).hexdigest()[:12])
    source_doc_id: Optional[str] = None
    start_char_idx: Optional[int] = None
    end_char_idx: Optional[int] = None

    def __repr__(self):
        return f"Node(id={self.id_}, text={self.text[:50]}...)"


# ============================================================================
# 手写版：NodeParser
# ============================================================================

class SimpleNodeParser:
    """
    手写版 NodeParser：按固定字符数切分

    原理：
    1. 把文本切成固定大小的块
    2. 相邻块之间保留重叠，避免切断语义
    3. 记录每个块在原文中的位置
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def get_nodes_from_documents(self, documents: List[Document]) -> List[Node]:
        nodes = []
        for doc in documents:
            text = doc.text
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end]

                node = Node(
                    text=chunk_text,
                    metadata={**doc.metadata, "chunk_index": len(nodes)},
                    source_doc_id=doc.id_,
                    start_char_idx=start,
                    end_char_idx=end,
                )
                nodes.append(node)

                # 步进 = chunk_size - overlap，确保相邻块有重叠
                start += self.chunk_size - self.chunk_overlap

        return nodes


# ============================================================================
# 手写版：Embedding 模型
# ============================================================================

class MockEmbeddingModel:
    """
    手写版 Embedding 模型（模拟）

    真实实现：调用 OpenAI / Weelink API
    这里用简化版：基于字符频率统计生成向量，用于展示原理

    原理：
    1. 统计文本中每个字符的出现频率
    2. 将频率向量映射到固定维度（如 384 维）
    3. 归一化，使所有向量长度相同
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        # 预定义字符集合，用于生成固定维度的向量
        self.vocab = [chr(i) for i in range(32, 127)]  # 可打印 ASCII

    def get_text_embedding(self, text: str) -> np.ndarray:
        """把文本转成固定维度的向量"""
        # 1. 统计字符频率
        freq = np.zeros(len(self.vocab))
        for char in text.lower():
            if char in self.vocab:
                idx = self.vocab.index(char)
                freq[idx] += 1

        # 2. 映射到目标维度（通过哈希组合）
        vec = np.zeros(self.dim)
        for i in range(self.dim):
            # 用多个频率值的组合生成每个维度
            source_idx = i % len(self.vocab)
            vec[i] = freq[source_idx] * (1 + i / self.dim)

        # 3. 归一化（L2 归一化，使向量长度为 1）
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec

    def get_text_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        return [self.get_text_embedding(t) for t in texts]


class WeelinkEmbeddingModel:
    """真实版：调用 Weelink API 做 Embedding"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def get_text_embedding(self, text: str) -> np.ndarray:
        import requests
        resp = requests.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": text},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return np.array(data["data"][0]["embedding"])

    def get_text_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        return [self.get_text_embedding(t) for t in texts]


# ============================================================================
# 手写版：VectorStoreIndex
# ============================================================================

class VectorStoreIndex:
    """
    手写版 VectorStoreIndex：向量索引

    原理：
    1. 把每个 Node 的文本转成向量
    2. 存储 Node 和向量
    3. 查询时：把查询文本也转成向量，计算余弦相似度
    4. 返回 Top-K 最相似的 Node

    相似度计算：余弦相似度 = dot(A, B) / (|A| * |B|)
    由于向量已归一化，简化为：dot(A, B)
    """

    def __init__(self, nodes: List[Node], embed_model):
        self.nodes = {n.id_: n for n in nodes}
        self.embed_model = embed_model

        # 预计算所有 Node 的向量
        print(f"  [Index] 正在向量化 {len(nodes)} 个 Node...")
        texts = [n.text for n in nodes]
        self.vectors = embed_model.get_text_embeddings(texts)
        print(f"  [Index] 向量化完成，维度: {len(self.vectors[0])}")

    def similarity_search(self, query: str, top_k: int = 3) -> List[Tuple[Node, float]]:
        """检索：计算查询向量与所有 Node 向量的相似度"""
        query_vec = self.embed_model.get_text_embedding(query)

        # 计算余弦相似度（向量已归一化，直接用点积）
        similarities = []
        for i, vec in enumerate(self.vectors):
            sim = float(np.dot(query_vec, vec))
            similarities.append((i, sim))

        # 按相似度排序，取 Top-K
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = min(top_k, len(similarities))

        results = []
        for idx, score in similarities[:top_k]:
            node_id = list(self.nodes.keys())[idx]
            results.append((self.nodes[node_id], score))

        return results


# ============================================================================
# 手写版：Query Engine
# ============================================================================

class WeelinkLLM:
    """真实版：调用 Weelink API 做 LLM 生成"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def complete(self, prompt: str) -> str:
        import requests
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 2048,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


class QueryEngine:
    """
    手写版 Query Engine：检索 + 生成

    原理：
    1. 用 Index 检索 Top-K 相关 Node
    2. 把检索内容拼成 prompt
    3. 调用 LLM 生成答案
    """

    def __init__(self, index: VectorStoreIndex, llm: WeelinkLLM, top_k: int = 3):
        self.index = index
        self.llm = llm
        self.top_k = top_k

    def query(self, question: str) -> Dict:
        # 1. 检索
        results = self.index.similarity_search(question, top_k=self.top_k)

        # 2. 构建 prompt
        context_parts = []
        for i, (node, score) in enumerate(results):
            context_parts.append(f"[段落 {i+1}] (相似度: {score:.4f})\n{node.text}")
        context = "\n\n".join(context_parts)

        prompt = f"""基于以下检索内容回答问题。如果检索内容不足以回答，请说明。

检索内容：
{context}

问题：{question}

请给出准确、简洁的回答。"""

        # 3. 生成
        answer = self.llm.complete(prompt)

        return {
            "query": question,
            "answer": answer,
            "sources": [
                {
                    "text": node.text[:200],
                    "score": score,
                    "metadata": node.metadata,
                }
                for node, score in results
            ],
        }


# ============================================================================
# 手写版：Reader
# ============================================================================

class SimpleDirectoryReader:
    """手写版 Reader：从目录加载 Markdown 文件"""

    def __init__(self, input_dir: str, required_exts: List[str] = None):
        self.input_dir = Path(input_dir)
        self.required_exts = required_exts or [".md"]

    def load_data(self) -> List[Document]:
        documents = []
        for ext in self.required_exts:
            for filepath in self.input_dir.rglob(f"*{ext}"):
                text = filepath.read_text(encoding="utf-8")
                doc = Document(
                    text=text,
                    metadata={"file_name": filepath.name, "file_path": str(filepath)},
                )
                documents.append(doc)
        return documents


# ============================================================================
# 对比实验
# ============================================================================

def experiment1_handmade_vs_official():
    """
    实验 1：手写版 vs 官方版对比

    展示手写版的核心原理，与 LlamaIndex 官方版的差异。
    """
    print("\n" + "=" * 70)
    print("实验 1：手写版 vs 官方版对比")
    print("=" * 70)

    print("""
手写版 vs 官方版的核心差异：

┌─────────────────┬────────────────────────────┬────────────────────────────┐
│     组件        │         手写版              │         官方版              │
├─────────────────┼────────────────────────────┼────────────────────────────┤
│ Document        │  dataclass (text + meta)    │  Document 类，支持更多元数据  │
│ Node            │  dataclass (text + meta)    │  Node 类，支持关系图、层级  │
│ NodeParser      │  按字符切分                  │  多种策略（Semantic/层次等）  │
│ Embedding       │  Mock（字符频率统计）        │  调用 OpenAI/Weelink API    │
│ VectorStore     │  内存中的 numpy 数组          │  可插拔（Qdrant/Pinecone等）│
│ 相似度计算      │  暴力 dot product            │  ANN 近似最近邻（HNSW等）   │
│ Query Engine    │  简单 prompt 拼接            │  多种合成模式（compact等）  │
│ 引用追踪        │  手动记录 score              │  自动 source_nodes 追踪     │
└─────────────────┴────────────────────────────┴────────────────────────────┘

关键结论：
- 手写版展示的是"核心原理"，不是"生产代码"
- 官方版的差异在于：可扩展性、性能、多种策略、错误处理
- 理解手写版后，看官方版源码会更清晰
""")


def experiment2_run_handmade_rag():
    """
    实验 2：运行手写版完整 RAG

    用共享数据跑通：Reader → Parser → Index → Query Engine
    """
    print("\n" + "=" * 70)
    print("实验 2：运行手写版完整 RAG")
    print("=" * 70)

    # 1. 读取环境变量
    api_key = os.environ.get("WEELINK_API_KEY")
    base_url = os.environ.get("WEELINK_BASE_URL")
    llm_model = os.environ.get("WEELINK_LLM_MODEL", "deepseek-v4-flash")
    embed_model_name = os.environ.get("WEELINK_EMBED_MODEL", "qwen3-embedding-8b")

    if not api_key:
        print("⚠️  未配置 WEELINK_API_KEY，使用 Mock Embedding 演示...")
        use_mock = True
    else:
        use_mock = False

    # 2. Reader
    print("\n[Step 1] Reader — 加载文档...")
    data_dir = Path(__file__).parent.parent.parent / "shared-data"
    reader = SimpleDirectoryReader(str(data_dir), required_exts=[".md"])
    documents = reader.load_data()
    print(f"  加载了 {len(documents)} 个文档")
    for d in documents:
        print(f"    - {d.metadata['file_name']} ({len(d.text)} chars)")

    # 3. NodeParser
    print("\n[Step 2] NodeParser — 切分文档...")
    parser = SimpleNodeParser(chunk_size=512, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(documents)
    print(f"  切分为 {len(nodes)} 个 Node")
    print(f"  Node 1 示例: {nodes[0].text[:100]}...")

    # 4. Embedding + Index
    print("\n[Step 3] Index — 构建向量索引...")
    if use_mock:
        embed_model = MockEmbeddingModel(dim=384)
        print("  使用 Mock Embedding（字符频率统计）")
    else:
        embed_model = WeelinkEmbeddingModel(api_key, base_url, embed_model_name)
        print(f"  使用 Weelink Embedding: {embed_model_name}")

    index = VectorStoreIndex(nodes, embed_model)

    # 5. Query Engine
    print("\n[Step 4] Query Engine — 查询...")
    if use_mock:
        print("  ⚠️  Mock Embedding 无法进行语义检索，跳过 LLM 查询")
        print("  演示检索过程：")
        results = index.similarity_search("Transformer 注意力机制", top_k=3)
        for i, (node, score) in enumerate(results):
            print(f"    [{i+1}] score={score:.4f} | {node.text[:80]}...")
    else:
        llm = WeelinkLLM(api_key, base_url, llm_model)
        engine = QueryEngine(index, llm, top_k=3)

        queries = [
            "Transformer 的核心机制是什么？",
            "RAG 为什么能解决大模型的幻觉问题？",
        ]

        for query in queries:
            print(f"\n  查询: {query}")
            print("  " + "-" * 50)
            result = engine.query(query)
            print(f"  答案: {result['answer'][:200]}...")
            print(f"  来源:")
            for i, src in enumerate(result['sources']):
                print(f"    [{i+1}] score={src['score']:.4f} | {src['text'][:60]}...")


def experiment3_understand_the_math():
    """
    实验 3：理解数学原理

    展示向量检索的核心数学：余弦相似度
    """
    print("\n" + "=" * 70)
    print("实验 3：理解向量检索的数学原理")
    print("=" * 70)

    print("""
核心公式：余弦相似度（Cosine Similarity）

    cos(θ) = (A · B) / (|A| × |B|)

其中：
    - A · B = dot product = Σ(Ai × Bi)
    - |A| = sqrt(Σ(Ai²))  （向量的 L2 范数）

简化：如果向量已经归一化（|A| = |B| = 1），则：
    cos(θ) = A · B

示例：
""")

    # 创建两个简单的向量
    A = np.array([1.0, 0.0, 0.0])  # 代表 "苹果"
    B = np.array([0.9, 0.1, 0.0])  # 代表 "水果"
    C = np.array([0.0, 1.0, 0.0])  # 代表 "汽车"

    # 归一化
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)
    C = C / np.linalg.norm(C)

    print(f"向量 A (苹果): {A}")
    print(f"向量 B (水果): {B}")
    print(f"向量 C (汽车): {C}")
    print()
    print(f"A · B = {np.dot(A, B):.4f}  ← 苹果和水果：相似度高")
    print(f"A · C = {np.dot(A, C):.4f}  ← 苹果和汽车：相似度低")
    print()
    print("结论：余弦相似度范围 [-1, 1]，值越大表示越相似。")
    print("      在 RAG 中，我们用它来找出与查询最相关的文本块。")


def experiment4_performance_comparison():
    """
    实验 4：性能对比

    展示暴力检索 vs 近似最近邻（ANN）的性能差异
    """
    print("\n" + "=" * 70)
    print("实验 4：暴力检索 vs 近似最近邻（ANN）")
    print("=" * 70)

    print("""
暴力检索（手写版）：
  - 计算查询向量与所有 N 个向量的点积
  - 时间复杂度: O(N × D)  （N=向量数，D=维度）
  - 10万条数据: 可接受
  - 100万条数据: 明显变慢
  - 10亿条数据: 不可行

近似最近邻（ANN，如 HNSW）：
  - 构建图结构，快速定位相似区域
  - 查询复杂度: O(log N)
  - 牺牲少量精度（recall ≈ 95-99%）
  - 可处理十亿级数据

对比：
┌────────────┬──────────────┬──────────────┬──────────────┐
│  数据规模   │  暴力检索     │  HNSW (ANN)  │  延迟感受     │
├────────────┼──────────────┼──────────────┼──────────────┤
│ 1,000      │  1 ms        │  0.1 ms      │  无差异       │
│ 100,000    │  50 ms       │  1 ms        │  可感知       │
│ 10,000,000 │  5,000 ms    │  5 ms        │  天壤之别     │
└────────────┴──────────────┴──────────────┴──────────────┘
""")


def main():
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "LlamaIndex 阶段四：手写简化版" + " " * 10 + "║")
    print("╚" + "═" * 68 + "╝")

    experiment1_handmade_vs_official()
    experiment2_run_handmade_rag()
    experiment3_understand_the_math()
    experiment4_performance_comparison()

    print("\n" + "=" * 70)
    print("阶段四完成！")
    print("=" * 70)
    print("""
你现在已经掌握了：

  ✅ Document / Node 的核心抽象
  ✅ NodeParser 的切分原理（chunk_size + overlap）
  ✅ Embedding 的本质（文本 → 向量）
  ✅ VectorStoreIndex 的检索原理（余弦相似度）
  ✅ Query Engine 的生成流程（检索 + prompt + LLM）
  ✅ 暴力检索 vs ANN 的性能差异

手写版的核心代码只有 ~200 行，但展示了 RAG 的全部原理。
官方 LlamaIndex 的复杂之处在于：
  - 可插拔设计（多种存储、多种模型）
  - 生产级特性（容错、缓存、流式）
  - 丰富的策略选择（多种 chunking、索引、重排序）

建议下一步：阅读官方 LlamaIndex 源码，对比手写版，理解设计决策。
""")


if __name__ == "__main__":
    main()
