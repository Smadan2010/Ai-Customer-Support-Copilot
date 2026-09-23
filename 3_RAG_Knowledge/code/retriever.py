"""Reusable semantic retriever that returns source-grounded ZENDS chunks."""

from __future__ import annotations

import re
from pathlib import Path

from embeddings import SentenceTransformerEmbedder
from vector_store import ZendsVectorStore


DEFAULT_TOP_K = 4
ENTITY_EXPANSIONS = (
    (("prepaid", "postpaid", "mobile"), ("prepaid", "postpaid", "mobile connectivity")),
    (("zendfiber", "zendoffice", "broadband", "home internet", "office internet"), ("zendfiber", "zendoffice")),
    (("zendbiz", "zendenterprise", "business connectivity"), ("zendbiz", "zendenterprise")),
    (("zendcloud", "zendstorage", "zendarchive", "cloud"), ("zendcloud", "zendstorage", "zendarchive")),
    (("zendsmart", "zendindustrial", "zendfleet", "iot"), ("zendsmart", "zendindustrial", "zendfleet")),
)


class ZendsRetriever:
    """Retrieve relevant source chunks only; this component never generates an answer."""

    def __init__(self, database_path: str | Path, *, top_k: int = DEFAULT_TOP_K, embedder: SentenceTransformerEmbedder | None = None) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        self.store = ZendsVectorStore(database_path)
        self.top_k = top_k
        self.embedder = embedder or SentenceTransformerEmbedder()

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        policy_category: str | None = None,
        support_intent: bool = False,
        product_query: bool = False,
    ) -> list[dict]:
        if not isinstance(query, str):
            raise TypeError("Retriever query must be a string.")
        cleaned = " ".join(query.split())
        if not cleaned:
            raise ValueError("Retriever query cannot be empty.")
        embedding = self.embedder.encode([cleaned])[0]
        # Support questions use a slightly wider candidate pool only for
        # evidence discovery. Segment 4 still selects at most two focused
        # chunks before answer generation.
        candidate_count = top_k or (8 if support_intent or product_query else self.top_k)
        result = self.store.query(embedding, candidate_count)
        chunks = [
            {"text": document, "metadata": metadata, "distance": float(distance)}
            for document, metadata, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
        ]
        # An explicit policy heading is stronger than broad semantic similarity.
        # Retain the regular retrieval results for traceability, but guarantee
        # that a matching policy section reaches evidence selection when present.
        if policy_category:
            policy_result = self.store.query(embedding, 1, where={"policy_category": policy_category})
            policy_chunks = [
                {"text": document, "metadata": metadata, "distance": float(distance)}
                for document, metadata, distance in zip(
                    policy_result["documents"][0], policy_result["metadatas"][0], policy_result["distances"][0]
                )
            ]
            policy_ids = {str(chunk["metadata"].get("chunk_id")) for chunk in policy_chunks}
            chunks = policy_chunks + [chunk for chunk in chunks if str(chunk["metadata"].get("chunk_id")) not in policy_ids]
        if product_query:
            normalized_query = " ".join(re.findall(r"[a-z0-9]+", cleaned.lower()))
            evidence_terms = {
                evidence
                for triggers, evidence_values in ENTITY_EXPANSIONS
                if any(trigger in normalized_query for trigger in triggers)
                for evidence in evidence_values
            }
            feature_requested = any(
                term in normalized_query
                for term in ("feature", "include", "offer", "provide", "service", "solution")
            )
            existing_ids = {str(chunk["metadata"].get("chunk_id")) for chunk in chunks}
            records = self.store.all()
            for document, metadata, vector in zip(records["documents"], records["metadatas"], records["embeddings"]):
                chunk_id = str(metadata.get("chunk_id"))
                if chunk_id in existing_ids:
                    continue
                document_lower = str(document).lower()
                entity_match = any(term in document_lower for term in evidence_terms)
                capability_match = feature_requested and document_lower.lstrip().startswith("services include")
                if not (entity_match or capability_match):
                    continue
                cosine_distance = 1.0 - sum(float(left) * float(right) for left, right in zip(embedding, vector))
                chunks.append({"text": document, "metadata": metadata, "distance": float(cosine_distance)})
                existing_ids.add(chunk_id)
        return chunks
