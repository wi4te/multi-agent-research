"""Basic tests for the multi-agent research assistant."""
import os
import sys
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that core modules can be imported."""
    try:
        from agents.planner import planner_node
        from agents.researcher import researcher_node
        from agents.writer import writer_node
        from agents.critic import critic_node
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import agent modules: {e}")


def test_vector_search_mock():
    """Test mock embedding generation."""
    from tools.vector_search import _mock_embedding
    embedding = _mock_embedding("test text")
    assert isinstance(embedding, list)
    assert len(embedding) == 1536  # Must match Bedrock Titan dimensions
    assert all(isinstance(x, float) for x in embedding)


def test_mock_embedding_deterministic():
    """Same text should produce same embedding."""
    from tools.vector_search import _mock_embedding
    emb1 = _mock_embedding("hello world")
    emb2 = _mock_embedding("hello world")
    assert emb1 == emb2


def test_mock_embedding_different_texts():
    """Different texts should produce different embeddings."""
    from tools.vector_search import _mock_embedding
    emb1 = _mock_embedding("hello")
    emb2 = _mock_embedding("world")
    assert emb1 != emb2


def test_planner_node_mock():
    """Test planner generates a plan in mock mode."""
    os.environ["MOCK_MODE"] = "true"
    from agents.planner import planner_node
    state = {"question": "What is AI?"}
    result = planner_node(state)
    assert "plan" in result
    assert isinstance(result["plan"], list)
    assert len(result["plan"]) > 0


def test_critic_node_mock():
    """Test critic reviews an answer in mock mode."""
    os.environ["MOCK_MODE"] = "true"
    from agents.critic import critic_node
    state = {
        "question": "What is AI?",
        "draft_answer": "AI is artificial intelligence."
    }
    result = critic_node(state)
    assert "final_answer" in result
