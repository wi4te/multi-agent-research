"""Database connection and initialization."""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:postgres@db:5432/research")

# Use psycopg (v3) driver explicitly
engine = create_engine(DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database with pgvector extension and tables."""
    import time
    max_retries = 30
    retry_delay = 2
    
    # Wait for database to be ready
    for attempt in range(max_retries):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print(f"Database is ready (attempt {attempt + 1})")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Waiting for database... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"Database connection failed after {max_retries} attempts: {e}")
                raise
    
    # Enable pgvector extension and create tables
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    
    from database.models import Document, Conversation
    Base.metadata.create_all(bind=engine)
    print("Database initialized")
