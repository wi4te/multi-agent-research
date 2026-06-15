"""Planner agent: breaks down complex questions into sub-tasks."""
import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"


def get_llm():
    if MOCK_MODE:
        return None
    from langchain_aws import ChatBedrock
    return ChatBedrock(
        model_id=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        model_kwargs={"temperature": 0.3, "max_tokens": 1024}
    )


def mock_plan(question: str) -> list:
    """Generate a simple mock plan for testing."""
    return [
        f"Define the key concepts in: {question}",
        f"Research current approaches to: {question}",
        f"Identify practical applications of: {question}"
    ]


def planner_node(state: dict) -> dict:
    """Planner agent: decomposes the question."""
    if MOCK_MODE:
        return {"plan": mock_plan(state["question"])}

    from langchain.prompts import ChatPromptTemplate
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Break the question into 3-5 sub-tasks. Output ONLY a JSON list of strings."),
        ("human", "Question: {question}\n\nSub-tasks:")
    ])
    chain = prompt | llm
    response = chain.invoke({"question": state["question"]})

    text = response.content
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            plan = json.loads(match.group())
        except json.JSONDecodeError:
            plan = [s.strip().strip('"') for s in text.strip('[]').split(',') if s.strip()]
    else:
        plan = [state["question"]]

    return {"plan": plan}
