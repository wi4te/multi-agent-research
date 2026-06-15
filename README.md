# Multi-Agent Research Assistant

A production-grade AI research assistant using **LangChain**, **LangGraph**, **AWS Bedrock**, and **pgvector** for RAG (Retrieval-Augmented Generation).

## Architecture

- **FastAPI** — REST API server
- **LangGraph** — Multi-agent orchestration
- **4 Specialized Agents** — Planner, Researcher, Writer, Critic
- **AWS Bedrock (Claude)** — LLM for all agents
- **pgvector** — Vector database for RAG
- **Docker** — Containerization
- **AWS EC2** — Cloud hosting
- **GitHub Actions + ECR** — CI/CD pipeline

## Features

- Multi-agent research workflow with iterative refinement
- RAG-powered responses using vector similarity search
- Mock mode for testing without AWS credentials
- Docker Compose for easy local development
- Automated CI/CD deployment to AWS

## Local Development

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/multi-agent-research.git
cd multi-agent-research

# Copy environment template
cp .env.example .env
# Edit .env with your settings

# Run with Docker Compose
docker compose up --build
```

## API Endpoints

- `GET /` — Health check
- `POST /research` — Submit research question
- `POST /documents` — Add document to knowledge base
- `GET /search?q=...` — Search vector database

## Deployment

Automatic deployment via GitHub Actions on push to main branch.

## License

MIT
