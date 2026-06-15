"""LangGraph workflow: orchestrates the multi-agent collaboration."""
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from agents.planner import planner_node
from agents.researcher import researcher_node
from agents.writer import writer_node
from agents.critic import critic_node, should_revise


class ResearchState(TypedDict):
    """State shared between agents."""
    question: str
    plan: List[str]
    research_findings: str
    draft_answer: str
    critique: str
    final_answer: str
    iterations: int
    max_iterations: int


def create_research_workflow():
    """
    Build the multi-agent research workflow.
    
    Flow:
    START → Planner → Researcher → Writer → Critic
                                              ↓
                                    [approved/revise]
                                              ↓
                                         END/Writer
    """
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critic_node)
    
    # Define edges
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "critic")
    
    # Conditional edge: critic decides
    workflow.add_conditional_edges(
        "critic",
        should_revise,
        {
            "revise": "writer",  # loop back to revise
            "end": END
        }
    )
    
    return workflow.compile()
