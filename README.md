# AgentKnowledgeOS

- 作者：`zhicong992-droid`
- 版权：`Copyright © 2026 zhicong992-droid. All rights reserved.`
- 版本：`2.0.0`
- 更新时间：`2026-05-19`

AgentKnowledgeOS 是一套围绕企业知识资产整理、检索和追问设计的多智能体系统。它把文档处理、结构化抽取、图谱检索、上下文压缩与长期记忆放进同一条执行链路里，目标不是只做“存文档”，而是提供可追溯的知识消费入口。

## 这个项目解决什么问题

- 把非结构化文档转成可查询的知识片段
- 同时利用向量召回和关系图检索支撑复杂问题
- 在回答前压缩上下文，降低噪声和 token 浪费
- 为会话和业务范围保存可版本化的长期记忆
- 在索引更新时保留增量处理能力，而不是每次全量重建

## 处理链路

`Ingest -> Parse -> Extract -> Index -> Retrieve -> Compress -> Answer -> Remember`

系统里最关键的几层能力：

- `GraphRAG`：把语义召回和图关系查询组合起来
- `Context Compression`：对检索结果做裁剪、去重和信号筛选
- `Memory Store`：把长期记忆写入 Postgres 并支持 TTL、版本和 GC
- `LangGraph Workflow`：统一编排导入、问答、更新三类执行路径

## 当前实现

- 语言与运行时：Python
- 接口层：FastAPI
- 编排层：LangGraph
- 检索层：ChromaDB 或 PGVector + Neo4j
- 记忆层：Postgres
- 更新链路：Kafka / CDC

## 目录概览

- `python/api`：HTTP 入口和管理接口
- `python/agents`：解析、抽取、问答、更新等智能体实现
- `python/orchestrator`：工作流图定义
- `python/services`：图谱、向量、压缩、记忆等服务

## 启动方式

```bash
cd python
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8080
```

建议至少配置这些变量：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `PGVECTOR_DSN`
- `MEMORY_DSN`
- `KAFKA_BOOTSTRAP_SERVERS`

示例模板：

```env
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
VECTOR_STORE_TYPE=chroma
CHROMA_HOST=localhost
CHROMA_PORT=8000
PGVECTOR_DSN=postgresql://postgres:postgres@localhost:5432/knowledge
MEMORY_DSN=postgresql://postgres:postgres@localhost:5432/knowledge
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
API_HOST=0.0.0.0
API_PORT=8080
```

## 对外接口

文档导入：

- `POST /api/ingest/upload`
- `POST /api/ingest/batch`

知识问答：

- `POST /api/qa/ask`

管理接口：

- `GET /api/health`
- `GET /api/admin/stats`
- `POST /api/admin/update`
- `POST /api/admin/memory`
- `GET /api/admin/memory/{scope}`
- `POST /api/admin/memory/gc`

## 运行特征

- 问答结果会回传来源上下文和推理步骤
- 向量层不可用时可以退回图谱路径和兜底检索
- 长期记忆支持按 `scope` 和 `query` 召回
- 导入、问答、更新走不同工作流，但共享整体服务能力

## 适用场景

- 企业知识库问答
- 文档驱动的调研与分析
- 需要保留引用来源的内部知识系统
- 用于展示 GraphRAG、上下文管理与长期记忆工程化能力的作品集项目

## 设计取舍

- 选择 `GraphRAG`，是因为很多问题并不只是语义相似，还依赖关系路径
- 增加压缩层，是为了防止“检索到了很多，但真正有效信息很少”
- 记忆单独落库，是为了把问答上下文与可复用业务知识区分开
- 保留降级路径，是为了在依赖不稳定时仍然给出可解释结果
