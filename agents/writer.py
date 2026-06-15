"""Writer agent: composes the final answer from research findings."""
import os
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
        model_kwargs={"temperature": 0.5, "max_tokens": 2048}
    )


def writer_node(state: dict) -> dict:
    """Writer agent: drafts the answer."""
    if MOCK_MODE:
        draft = (
            f"# {state['question']}\n\n"
            f"Based on the research, here is the answer:\n\n"
            f"{state['research_findings']}\n\n"
            f"---\n*This is a mock answer for testing. In production, Claude would compose a polished response.*"
        )
        return {"draft_answer": draft}

    from langchain.prompts import ChatPromptTemplate
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a writer agent. Compose a clear, well-structured answer with headings and bullets."),
        ("human", "Question: {question}\nFindings: {findings}\nCritique: {critique}\n\nAnswer:")
    ])
    chain = prompt | llm
    response = chain.invoke({
        "question": state["question"],
        "findings": state["research_findings"],
        "critique": state.get("critique", "None")
    })
    return {"draft_answer": response.content}
