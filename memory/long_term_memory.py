"""
memory/long_term_memory.py
--------------------------
Long-term (external) memory using ChromaDB + sentence-transformers.

How it differs from RAG:
  - RAG         → retrieves relevant *document chunks* given a query.
  - Long-term memory → retrieves relevant *past agent sessions* (query + response pairs).
  Same mechanism (embedding similarity search), different data.

TTL (time-to-live):
  Entries older than `ttl_days` are silently excluded from retrieval.
  ChromaDB doesn't natively support TTL, so we filter on the `timestamp` metadata field.
"""

import uuid
import chromadb
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer


def _safe_collection_name(name: str) -> str:
    """
    Sanitise a string to meet ChromaDB collection naming rules:
      - 3-63 characters
      - Starts and ends with an alphanumeric character
      - Contains only alphanumerics, underscores, or hyphens
    """
    import re
    # Replace any disallowed char with underscore
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    # Strip leading/trailing underscores/hyphens so it starts/ends alphanumeric
    cleaned = cleaned.strip("_-")
    # Ensure minimum length of 3
    if len(cleaned) < 3:
        cleaned = (cleaned + "mem")[:3]
    # Enforce max 63 chars
    cleaned = cleaned[:63]
    # Final guard: if still starts/ends with non-alphanumeric, prepend/append 'x'
    if not cleaned[0].isalnum():
        cleaned = "x" + cleaned[1:]
    if not cleaned[-1].isalnum():
        cleaned = cleaned[:-1] + "x"
    return cleaned


class LongTermMemory:
    """
    Persistent ChromaDB-backed semantic memory.

    store()    — embed query+response and persist.
    retrieve() — semantic search; respects TTL.
    clear()    — wipe and recreate the collection (safe to call at any time).
    """

    def __init__(self, collection_name: str = "research_memory"):
        self._collection_name = _safe_collection_name(collection_name)

        # Persistent client stores data on disk under ./chroma_db
        self._client = chromadb.PersistentClient(path="./chroma_db")

        # get_or_create so re-instantiation never crashes
        self.collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )

        # Local embedding model (no API key needed)
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # ------------------------------------------------------------------

    def store(
        self,
        session_id: str,
        query: str,
        response: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Embed and persist a past session.

        Parameters
        ----------
        session_id : str  — unique identifier for this research run
        query      : str  — the topic / user query
        response   : str  — the agent's final output / file path
        metadata   : dict — any extra metadata to store alongside the entry
        """
        text      = f"Query: {query}\nResponse: {response}"
        embedding = self._embedder.encode(text).tolist()

        doc_meta = {
            "session_id": session_id,
            "timestamp":  datetime.utcnow().isoformat(),
            **(metadata or {}),
        }

        self.collection.add(
            ids        = [str(uuid.uuid4())],
            embeddings = [embedding],
            documents  = [text],
            metadatas  = [doc_meta],
        )

    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3, ttl_days: int = 7) -> list[str]:
        """
        Semantic search over stored sessions.

        Returns the text of the top-k matches that are newer than `ttl_days`.

        Why TTL matters:
          Without TTL, stale facts (e.g. "GPT-4 is the latest model") would be
          injected as if they were current, potentially misleading the agent.
        """
        total = self.collection.count()
        if total == 0:
            return []

        # Query at most `total` results to avoid n_results > collection size error
        n = min(top_k, total)

        embedding = self._embedder.encode(query).tolist()

        results = self.collection.query(
            query_embeddings = [embedding],
            n_results        = n,
            include          = ["documents", "metadatas"],
        )

        filtered: list[str] = []
        cutoff = datetime.utcnow() - timedelta(days=ttl_days)

        if results.get("documents"):
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                ts_str = meta.get("timestamp", "")
                try:
                    ts = datetime.fromisoformat(ts_str)
                    if ts < cutoff:
                        continue   # skip stale entries (TTL expired)
                except ValueError:
                    pass   # if timestamp is unparseable, include anyway

                filtered.append(doc)

        return filtered

    # ------------------------------------------------------------------

    def clear(self) -> None:
        """
        Wipe the entire collection and recreate it (safe to call any time).

        Bug fix: the original code deleted but never recreated the collection,
        causing AttributeError on the next store/retrieve call.
        """
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass  # collection might not exist — that's fine

        # Recreate immediately so the object remains usable
        self.collection = self._client.get_or_create_collection(
            name     = self._collection_name,
            metadata = {"hnsw:space": "cosine"},
        )