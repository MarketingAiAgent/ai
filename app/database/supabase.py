# app/integrations/supabase.py
from __future__ import annotations

import os
import logging
from typing import List, Optional

from supabase import create_client, Client  
from supabase.client import ClientOptions
from openai import OpenAI 

from app.core.config import settings 

logger = logging.getLogger(__name__)

# =============================================================================
# Embeddings Provider (OpenAI)
# =============================================================================

class OpenAIEmbeddings:
    """
    Simple embedding wrapper with the minimal API:
      - embed_query(text) -> List[float]
      - embed_documents(texts) -> List[List[float]]

    NOTE:
      - Ensure your Supabase/pgvector column dimension matches the chosen model.
        text-embedding-3-small: 1536 dims
        text-embedding-3-large: 3072 dims
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def embed_query(self, text: str) -> List[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("`text` must be a non-empty string.")
        resp = self._client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding 

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]  # type: ignore[attr-defined]


# =============================================================================
# Supabase Client
# =============================================================================

def _init_supabase_client() -> Optional["Client"]:
    url = settings.SUPERBASE_URL
    key = settings.SUPABASE_ANON_KEY

    if not url or not key:
        logger.error("Supabase not configured: set SUPABASE_URL and SUPABASE_ANON_KEY (or SERVICE_ROLE_KEY).")
        return None

    if create_client is None:
        raise RuntimeError(
            "supabase package is not available. Install with `pip install supabase`."
        )

    return create_client(url, key, options=ClientOptions(postgrest_client_timeout=60))

def _init_embeddings() -> Optional[OpenAIEmbeddings]:
    try:
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)
    except Exception as e:
        logger.warning("Embeddings not configured: %s", e)
        return None


supabase_client: Optional["Client"] = _init_supabase_client()
embeddings: Optional[OpenAIEmbeddings] = _init_embeddings()

__all__ = ["supabase_client", "embeddings", "OpenAIEmbeddings"]