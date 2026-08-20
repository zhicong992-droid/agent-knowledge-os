import pytest

pytest.importorskip("langchain_core")

from agents.qa_agent import QAAgent, QueryIntent, RetrievedContext


class _Response:
    content = "综合答案：[来源: policy] 和 [来源: faq]"


class _LLM:
    def __init__(self):
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = messages
        return _Response()


@pytest.mark.asyncio
async def test_generate_answer_uses_all_contexts_and_llm():
    agent = QAAgent.__new__(QAAgent)
    agent.llm = _LLM()
    contexts = [
        RetrievedContext("第一条证据", "policy", 0.9, "vector"),
        RetrievedContext("第二条证据", "faq", 0.8, "graph"),
    ]

    answer, reasoning = await agent._generate_answer("怎么申请？", contexts, QueryIntent.PROCEDURAL)

    assert answer.startswith("综合答案")
    prompt = agent.llm.messages[1].content
    assert "第一条证据" in prompt
    assert "第二条证据" in prompt
    assert "答案生成完成" in reasoning


@pytest.mark.asyncio
async def test_generate_answer_refuses_without_context():
    agent = QAAgent.__new__(QAAgent)
    agent.llm = _LLM()

    answer, reasoning = await agent._generate_answer("未知问题", [], QueryIntent.FACTOID)

    assert "没有足够信息" in answer
    assert agent.llm.messages is None
    assert any("拒绝生成" in step for step in reasoning)
