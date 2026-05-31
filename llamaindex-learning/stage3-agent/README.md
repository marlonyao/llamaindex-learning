# 阶段三：Agent 化

## 目标

从"单轮问答"升级到"多步推理 + 工具调用"。

包含 4 个实验：
1. **FunctionAgent** — Query Engine 包装为 Tool
2. **多源 RAG** — 多个 Query Engine 路由
3. **Workflow** — 事件驱动编排
4. **多步推理** — 检索 → 提取 → 生成

## 前置条件

```bash
cd ../..
pip install -r requirements.txt
# .env 已配置好
```

## 运行

```bash
cd stage3-agent
python src/stage3_agent.py
```

## 实验说明

### 实验 1：FunctionAgent

把 Query Engine 包装成 Tool，让 LLM 决定何时调用。理解：
- Agent 有"工具选择"能力，不是简单的顺序调用
- 工具元数据（name + description）影响 Agent 的选择
- FunctionAgent 使用原生 tool calling，比 ReActAgent 更可靠

### 实验 2：多源 RAG

按主题把文档分成多个索引，Agent 自动路由。理解：
- 不同领域的数据需要不同的处理策略
- Agent 的 routing 能力让多源查询成为可能
- 工具描述越清晰，路由越准确

### 实验 3：Workflow

事件驱动的多步编排。理解：
- `@step` 装饰器定义工作流节点
- 事件类型定义数据流
- 支持并行执行、循环、状态恢复

### 实验 4：多步推理

固定流程：先检索 → 再提取结构化数据 → 再生成报告。理解：
- 不是所有场景都需要 Agent 的灵活性
- 固定流程用 Workflow，灵活决策用 Agent
- 多步可以分解复杂任务，提高质量

## 思考题

1. FunctionAgent 和 ReActAgent 有什么区别？什么时候选哪个？
2. 多源 RAG 中，工具描述怎么写才能让 Agent 准确路由？
3. Workflow 和 Agent 的区别是什么？什么场景用 Workflow，什么用 Agent？
4. 多步推理中，如果中间步骤失败了，怎么回退或重试？

## 下一步

完成阶段三后，进入 [阶段四：手写简化版 LlamaIndex](../stage4-mini-llamaindex/README.md)
