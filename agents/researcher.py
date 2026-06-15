"""Researcher agent: gathers information from vector DB and web."""
import os
from dotenv import load_dotenv
from tools.vector_search import search_similar
from tools.web_search import web_search

load_dotenv()
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"


def get_llm():
    if MOCK_MODE:
        return None
    from langchain_aws import ChatBedrock
    return ChatBedrock(
        model_id=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        model_kwargs={"temperature": 0.3, "max_tokens": 2048}
    )


def mock_research(question: str, plan: list) -> str:
    """Generate mock research findings."""
    vector_results = search_similar(question, k=3)
    vec_text = "\n".join([
        f"- [{r['source']}] {r['content'][:200]}"
        for r in vector_results
    ]) or "No documents in knowledge base yet."

    return (
        f"Mock research findings for: {question}\n\n"
        f"Sub-tasks investigated: {plan}\n\n"
        f"From knowledge base:\n{vec_text}\n\n"
        f"Key insights: This is mock data for testing. In production, this would "
        f"contain synthesized information from Claude."
    )


def researcher_node(state: dict) -> dict:
    """Researcher agent: gathers info from multiple sources."""
    if MOCK_MODE:
        return {"research_findings": mock_research(state["question"], state["plan"])}

    from langchain.prompts import ChatPromptTemplate
    vector_results = search_similar(state["question"], k=5)
    web_results = web_search(state["question"])

    vec_text = "\n".join([f"- [{r['source']}] {r['content'][:300]}" for r in vector_results]) or "No vector DB results."
    web_text = "\n".join([f"- {r['title']}: {r['snippet']}" for r in web_results]) or "No web results."

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research agent. Synthesize findings from context. Be factual."),
        ("human", "Question: {question}\nPlan: {sub_tasks}\nVector: {vector_results}\nWeb: {web_results}\n\nFindings:")
    ])
    chain = prompt | llm
    response = chain.invoke({
        "question": state["question"],
        "sub_tasks": state["plan"],
        "vector_results": vec_text,
        "web_results": web_text
    })
    return {"research_findings": response.content}
