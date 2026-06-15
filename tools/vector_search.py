"""Vector search tool using pgvector on RDS."""
import os
from typing import List, Dict
from dotenv import load_dotenv
from sqlalchemy import text
from database.connection import SessionLocal

load_dotenv()
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"


def get_embeddings():
    """Initialize Bedrock Titan embeddings, or return mock in MOCK_MODE."""
    if MOCK_MODE:
        return None
    from langchain_aws import BedrockEmbeddings
    return BedrockEmbeddings(
        model_id=os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v1"),
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )


def _mock_embedding(text: str) -> list:
    """Generate a fake 1536-dim embedding based on text hash."""
    import hashlib
    h = hashlib.md5(text.encode()).hexdigest()
    # Create deterministic pseudo-embedding of 1536 dimensions
    base = [float(int(h[i:i+2], 16)) / 255.0 for i in range(0, 32, 2)]  # 16 values
    return base * 96  # 16 * 96 = 1536


def add_document(content: str, source: str = None) -> str:
    """Add a document to the vector database."""
    db = SessionLocal()
    try:
        if MOCK_MODE:
            embedding = _mock_embedding(content)
        else:
            embeddings = get_embeddings()
            embedding = embeddings.embed_query(content)

        db.execute(
            text("INSERT INTO documents (id, content, source, embedding) VALUES (gen_random_uuid(), :content, :source, CAST(:embedding AS vector))"),
            {"content": content, "source": source, "embedding": embedding}
        )
        db.commit()
        return "Document added"
    finally:
        db.close()


def search_similar(query: str, k: int = 5) -> List[Dict[str, str]]:
    """Search for documents similar to the query."""
    db = SessionLocal()
    try:
        if MOCK_MODE:
            query_embedding = _mock_embedding(query)
        else:
            embeddings = get_embeddings()
            query_embedding = embeddings.embed_query(query)

        results = db.execute(
            text("""
                SELECT content, source, embedding <=> CAST(:embedding AS vector) AS distance
                FROM documents
                ORDER BY distance
                LIMIT :k
            """),
            {"embedding": query_embedding, "k": k}
        ).fetchall()

        return [
            {"content": r[0], "source": r[1] or "unknown", "distance": float(r[2])}
            for r in results
        ]
    finally:
        db.close()
