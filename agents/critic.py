"""Critic agent: reviews the draft and decides if it needs improvement."""
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
        model_kwargs={"temperature": 0.2, "max_tokens": 1024}
    )


def critic_node(state: dict) -> dict:
    """Critic agent: reviews and decides on revisions."""
    if MOCK_MODE:
        # In mock mode, always approve after 1 iteration
        return {
            "critique": "Mock critique: The draft covers the main points. STATUS: APPROVED",
            "final_answer": state["draft_answer"],
            "iterations": state.get("iterations", 0) + 1
        }

    from langchain.prompts import ChatPromptTemplate
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a critic. Review the draft. End with STATUS: APPROVED or STATUS: REVISE."),
        ("human", "Question: {question}\nDraft: {draft}\n\nCritique:")
    ])
    chain = prompt | llm
    response = chain.invoke({
        "question": state["question"],
        "draft": state["draft_answer"]
    })

    critique_text = response.content
    approved = "STATUS: APPROVED" in critique_text.upper()

    return {
        "critique": critique_text,
        "final_answer": state["draft_answer"] if approved else "",
        "iterations": state.get("iterations", 0) + 1
    }


def should_revise(state: dict) -> str:
    """Decide next step: revise or end."""
    if state.get("final_answer"):
        return "end"
    if state.get("iterations", 0) >= state.get("max_iterations", 2):
        return "end"
    return "revise"
