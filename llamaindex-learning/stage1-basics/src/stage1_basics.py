#!/usr/bin/env python3
"""
阶段一：LlamaIndex 核心概念实践

使用 Weelink 作为 LLM + Embedding 提供商（OpenAI-compatible API）

运行方式：
    1. 复制 .env.example 为 .env，填入你的 API Key
    2. python src/stage1_basics.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 环境变量
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# 1. Reader — 从文件系统加载文档
from llama_index.core import SimpleDirectoryReader

# 2. Document / Node — 文档切分
from llama_index.core.node_parser import SimpleNodeParser

# 3. Index — 索引构建
from llama_index.core import VectorStoreIndex

# 4. Retriever — 检索器
from llama_index.core.retrievers import VectorIndexRetriever

# 5. Query Engine — 查询引擎

# 使用 OpenAI-compatible 方式连接 Weelink
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding


def setup_api():
    """检查 API 配置"""
    api_key = os.environ.get("WEELINK_API_KEY")
    base_url = os.environ.get("WEELINK_BASE_URL")
    llm_model = os.environ.get("WEELINK_LLM_MODEL", "deepseek-v4-flash")
    embed_model = os.environ.get("WEELINK_EMBED_MODEL", "qwen3-embedding-8b")
    
    if not api_key:
        print("⚠️  未设置 WEELINK_API_KEY，请检查 .env 文件")
        return None, None, None, None
    
    print(f"API 配置：")
    print(f"  - Base URL: {base_url}")
    print(f"  - LLM: {llm_model}")
    print(f"  - Embedding: {embed_model}")
    
    return api_key, base_url, llm_model, embed_model


def create_llm(api_key, base_url, model):
    """创建 Weelink LLM 实例"""
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
    """创建 Weelink Embedding 实例（OpenAI-compatible）"""
    return OpenAIEmbedding(
        model=model,
        api_base=base_url,
        api_key=api_key,
        embed_batch_size=10,
    )


def step1_reader():
    """步骤 1：Reader — 加载文档"""
    print("=" * 60)
    print("步骤 1：Reader — 加载文档")
    print("=" * 60)
    
    data_dir = Path(__file__).parent.parent.parent / "shared-data"
    
    reader = SimpleDirectoryReader(
        input_dir=str(data_dir),
        required_exts=[".md"],
        recursive=True,
    )
    
    documents = reader.load_data()
    
    print(f"加载了 {len(documents)} 个文档：")
    for doc in documents:
        print(f"  - {doc.metadata.get('file_name', 'unknown')}")
        print(f"    内容长度: {len(doc.text)} 字符")
    
    return documents


def step2_nodes(documents):
    """步骤 2：Document / Node — 文档切分"""
    print("\n" + "=" * 60)
    print("步骤 2：Document / Node — 文档切分")
    print("=" * 60)
    
    parser = SimpleNodeParser.from_defaults(
        chunk_size=512,
        chunk_overlap=50,
    )
    nodes = parser.get_nodes_from_documents(documents)
    
    print(f"\n[SimpleNodeParser] 切分结果：")
    print(f"  总 Node 数: {len(nodes)}")
    print(f"  前 3 个 Node 的文本长度：")
    for i, node in enumerate(nodes[:3]):
        print(f"    Node {i}: {len(node.text)} chars | 来源: {node.metadata.get('file_name', '?')}")
    
    return nodes


def step3_index(nodes, embed_model):
    """步骤 3：Index — 构建向量索引"""
    print("\n" + "=" * 60)
    print("步骤 3：Index — 构建向量索引")
    print("=" * 60)
    
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    
    print(f"索引构建完成！")
    print(f"  - 索引中的 Node 数: {len(nodes)}")
    
    return index


def step4_retriever(index):
    """步骤 4：Retriever — 检索器"""
    print("\n" + "=" * 60)
    print("步骤 4：Retriever — 测试检索")
    print("=" * 60)
    
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=3,
    )
    
    query = "Transformer 的注意力机制是怎么计算的？"
    print(f"\n查询: {query}")
    
    nodes = retriever.retrieve(query)
    
    print(f"\n检索到 {len(nodes)} 个相关 Node：")
    for i, node in enumerate(nodes):
        print(f"\n  Node {i+1} (相似度: {node.score:.4f})")
        print(f"  来源: {node.node.metadata.get('file_name', '?')}")
        print(f"  内容: {node.node.text[:200]}...")
    
    return retriever


def step5_query_engine(index, llm):
    """步骤 5：Query Engine — 完整 RAG 查询"""
    print("\n" + "=" * 60)
    print("步骤 5：Query Engine — 完整 RAG 查询")
    print("=" * 60)
    
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=3,
        response_mode="compact",
    )
    
    queries = [
        "Transformer 的核心机制是什么？",
        "LlamaIndex 和 LangChain 有什么区别？",
        "RAG 为什么能解决大模型的幻觉问题？",
    ]
    
    for query in queries:
        print(f"\n{'─' * 50}")
        print(f"查询: {query}")
        print(f"{'─' * 50}")
        
        response = query_engine.query(query)
        
        print(f"答案:\n{response.response}")
        print(f"\n来源 ({len(response.source_nodes)} 个)：")
        for node in response.source_nodes:
            print(f"  - {node.node.metadata.get('file_name', '?')} (score: {node.score:.4f})")
    
    return query_engine


def step6_show_nodes(index):
    """Bonus：查看索引中的 Node 详情"""
    print("\n" + "=" * 60)
    print("Bonus：索引中的 Node 元数据")
    print("=" * 60)
    
    nodes = index.docstore.docs.values()
    
    for node in list(nodes)[:3]:
        print(f"\nNode ID: {node.id_}")
        print(f"  来源: {node.metadata.get('file_name', '?')}")
        print(f"  文本长度: {len(node.text)} chars")
        print(f"  文本前 100 字: {node.text[:100]}...")


def main():
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "LlamaIndex 阶段一：核心概念实践" + " " * 10 + "║")
    print("║" + " " * 15 + "Provider: Weelink" + " " * 18 + "║")
    print("╚" + "═" * 58 + "╝")
    
    config = setup_api()
    if not config[0]:
        return
    
    api_key, base_url, llm_model, embed_model_name = config
    
    llm = create_llm(api_key, base_url, llm_model)
    embed_model = create_embed_model(api_key, base_url, embed_model_name)
    
    documents = step1_reader()
    nodes = step2_nodes(documents)
    index = step3_index(nodes, embed_model)
    retriever = step4_retriever(index)
    query_engine = step5_query_engine(index, llm)
    step6_show_nodes(index)
    
    print("\n" + "=" * 60)
    print("阶段一完成！")
    print("=" * 60)
    print("""
你现在已经理解了 LlamaIndex 的 5 个核心概念：

  ✅ Reader        — 从各种源加载数据
  ✅ Document/Node — 文档切分为检索单元
  ✅ Index         — 构建向量索引
  ✅ Retriever     — 检索相关节点
  ✅ Query Engine  — 完整的 RAG 查询流程

Provider: Weelink | LLM: {llm} | Embedding: {embed}

接下来进入阶段二：检索质量深度优化。
""".format(llm=llm_model, embed=embed_model_name))


if __name__ == "__main__":
    main()
