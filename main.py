"""
Main FastAPI application for Multi-Agent Research Assistant.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import logging

from graph.workflow import create_research_workflow
from database.connection import init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize workflow on startup
workflow = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and workflow on startup."""
    global workflow
    logger.info("Starting up...")
    init_db()
    workflow = create_research_workflow()
    logger.info("Workflow ready")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Multi-Agent Research Assistant",
    description="AI system using LangGraph for collaborative research",
    version="0.1.0",
    lifespan=lifespan
)


class QuestionRequest(BaseModel):
    question: str
    max_iterations: int = 2


class ResearchResponse(BaseModel):
    question: str
    plan: list
    research_findings: str
    draft_answer: str
    final_answer: str
    iterations: int
    status: str


@app.get("/")
def root():
    return {
        "service": "Multi-Agent Research Assistant",
        "status": "running",
        "agents": ["planner", "researcher", "writer", "critic"]
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/research", response_model=ResearchResponse)
def research(request: QuestionRequest):
    """
    Run the multi-agent research workflow on a question.
    """
    if not workflow:
        raise HTTPException(status_code=503, detail="Workflow not initialized")
    
    try:
        result = workflow.invoke({
            "question": request.question,
            "plan": [],
            "research_findings": "",
            "draft_answer": "",
            "critique": "",
            "final_answer": "",
            "iterations": 0,
            "max_iterations": request.max_iterations
        })
        
        return ResearchResponse(
            question=result["question"],
            plan=result["plan"],
            research_findings=result["research_findings"],
            draft_answer=result["draft_answer"],
            final_answer=result["final_answer"],
            iterations=result["iterations"],
            status="complete"
        )
    except Exception as e:
        logger.error(f"Research failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
