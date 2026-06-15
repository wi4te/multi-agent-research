"""Script to add sample documents to the vector database for testing."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.vector_search import add_document

# Sample knowledge base about AI/ML topics
SAMPLE_DOCS = [
    """Retrieval-Augmented Generation (RAG) is a technique that combines a retriever 
    with a generator. The retriever fetches relevant documents from a knowledge base, 
    and the generator uses those documents to produce more accurate and grounded 
    responses. RAG helps reduce hallucinations in LLMs.""",
    
    """LangGraph is a framework for building stateful, multi-agent applications with 
    LLMs. It uses a graph-based approach where nodes are agents or functions, and 
    edges define the flow of execution. LangGraph supports cycles, conditional 
    routing, and human-in-the-loop patterns.""",
    
    """Vector databases store high-dimensional embeddings and enable fast similarity 
    search. Popular vector databases include Pinecone, Weaviate, Milvus, and 
    pgvector (PostgreSQL extension). They use algorithms like HNSW or IVF for 
    approximate nearest neighbor search.""",
    
    """Amazon Bedrock is a fully managed service that provides access to foundation 
    models from leading AI companies including Anthropic (Claude), AI21 (Jurassic), 
    Cohere, and Meta (Llama) through a single API. It supports fine-tuning and 
    RAG with knowledge bases.""",
    
    """The Model Context Protocol (MCP) is an open standard for connecting AI 
    assistants to data sources and tools. It enables AI models to interact with 
    external systems in a standardized way, similar to how LSP works for code 
    editors.""",
    
    """LangChain is a framework for developing applications powered by language 
    models. It provides tools for chaining LLM calls, managing memory, handling 
    documents, and integrating with external data sources and APIs.""",
    
    """Multi-agent systems use multiple specialized AI agents that collaborate to 
    solve complex problems. Each agent has a specific role (e.g., planner, 
    researcher, writer, critic) and they communicate through shared state. This 
    approach improves quality through specialization and self-critique."""
]

if __name__ == "__main__":
    print("Adding sample documents to vector database...")
    for i, doc in enumerate(SAMPLE_DOCS, 1):
        try:
            add_document(doc, source=f"sample_doc_{i}")
            print(f"Added document {i}/{len(SAMPLE_DOCS)}")
        except Exception as e:
            print(f"Failed to add doc {i}: {e}")
    print("Done!")
