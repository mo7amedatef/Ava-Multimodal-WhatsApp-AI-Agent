import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import List, Optional

from ai_companion.settings import settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


@dataclass
class Memory:
    """Represents a memory entry in the vector store."""

    text: str
    metadata: dict
    score: Optional[float] = None

    @property
    def id(self) -> Optional[str]:
        return self.metadata.get("id")

    @property
    def timestamp(self) -> Optional[datetime]:
        ts = self.metadata.get("timestamp")
        return datetime.fromisoformat(ts) if ts else None


class VectorStore:
    """A class to handle vector storage operations using Qdrant."""

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    COLLECTION_NAME = "long_term_memory"
    SIMILARITY_THRESHOLD = 0.9

    def __init__(self) -> None:
        self.model = SentenceTransformer(self.EMBEDDING_MODEL)
        self.client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

    def _collection_exists(self) -> bool:
        """Check if the memory collection exists."""
        collections = self.client.get_collections().collections
        return any(col.name == self.COLLECTION_NAME for col in collections)

    def _create_collection(self) -> None:
        """Create a new collection for storing memories."""
        sample_embedding = self.model.encode("sample text")
        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=len(sample_embedding),
                distance=Distance.COSINE,
            ),
        )

    def _build_user_filter(self, user_id: str) -> Filter:
        """Build a Qdrant filter that scopes results to a single user."""
        return Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        )

    def find_similar_memory(self, text: str, user_id: str) -> Optional[Memory]:
        """Find if a similar memory already exists for this user.

        Args:
            text: The text to search for
            user_id: The user to scope the search to

        Returns:
            Optional Memory if a similar one is found
        """
        results = self.search_memories(text, user_id=user_id, k=1)
        if results and results[0].score >= self.SIMILARITY_THRESHOLD:
            return results[0]
        return None

    def store_memory(self, text: str, metadata: dict, user_id: str) -> None:
        """Store a new memory in the vector store, scoped to a specific user.

        Args:
            text: The text content of the memory
            metadata: Additional information about the memory (timestamp, type, etc.)
            user_id: The owner of this memory
        """
        if not self._collection_exists():
            self._create_collection()

        embedding = self.model.encode(text)
        point = PointStruct(
            id=metadata.get("id", hash(text)),
            vector=embedding.tolist(),
            payload={
                "text": text,
                "user_id": user_id,
                **metadata,
            },
        )

        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[point],
        )

    def search_memories(self, query: str, user_id: str, k: int = 5) -> List[Memory]:
        """Search for similar memories scoped to a specific user.

        Args:
            query: Text to search for
            user_id: The user to scope the search to
            k: Number of results to return

        Returns:
            List of Memory objects
        """
        if not self._collection_exists():
            return []

        query_embedding = self.model.encode(query)
        results = self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            query_filter=self._build_user_filter(user_id),
            limit=k,
        )

        return [
            Memory(
                text=hit.payload["text"],
                metadata={key: val for key, val in hit.payload.items() if key not in ("text", "user_id")},
                score=hit.score,
            )
            for hit in results
        ]


@lru_cache
def get_vector_store() -> VectorStore:
    """Get or create the VectorStore singleton instance."""
    return VectorStore()