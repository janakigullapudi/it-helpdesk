from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chromadb import PersistentClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from helpdesk.settings import get_settings, require_api_key
from helpdesk.telemetry import span

COLLECTION_NAME = "frozen_helpdesk_tickets_v1"


def _records() -> list[dict[str, str]]:
    path = get_settings().data_dir / "golden_subset.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"The frozen golden dataset is empty or invalid: {path}")
    for record in records:
        if set(record) != {"ticket_id", "initial_message", "resolution_summary"}:
            raise ValueError(f"Unexpected fields in frozen dataset: {path}")
    return records


def _embeddings() -> GoogleGenerativeAIEmbeddings:
    api_key = require_api_key()
    return GoogleGenerativeAIEmbeddings(
        model=get_settings().embedding_model,
        google_api_key=api_key,
    )


def search_knowledge(
    query: str,
    *,
    exclude_ticket_id: str | None = None,
    top_k: int | None = None,
    session_id: str | None = None,
    trace_id: str | None = None,
) -> list[dict[str, Any]]:
    records = _records()
    settings = get_settings()
    with span(
        "knowledge.index_and_search",
        agent_name="knowledge_search",
        session_id=session_id,
        trace_id=trace_id,
    ):
        try:
            client = PersistentClient(path=str(settings.chroma_dir))
        except ImportError as exc:
            if "chromadb_rust_bindings" not in str(exc):
                raise
            raise RuntimeError(
                "Chroma's Windows native binding could not load. Install the "
                "Microsoft Visual C++ 2015-2022 Redistributable (x64), then retry."
            ) from exc
        collection = client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        expected_ids = {record["ticket_id"] for record in records}
        indexed_ids = set(collection.get(include=["metadatas"])["ids"])
        if not indexed_ids:
            embedder = _embeddings()
            documents = [
                f"User issue: {record['initial_message']}\n"
                f"Resolution: {record['resolution_summary']}"
                for record in records
            ]
            vectors = embedder.embed_documents(documents)
            collection.add(
                ids=[record["ticket_id"] for record in records],
                documents=documents,
                metadatas=[{"ticket_id": record["ticket_id"]} for record in records],
                embeddings=vectors,
            )
        elif indexed_ids != expected_ids:
            raise RuntimeError(
                "Chroma index does not match the frozen dataset. Remove the "
                "configured Chroma directory and rerun to rebuild it."
            )

        eligible_count = len(expected_ids) - int(
            exclude_ticket_id in expected_ids if exclude_ticket_id else False
        )
        if eligible_count <= 0:
            return []
        embedder = _embeddings()
        query_vector = embedder.embed_query(query)
        filters = {"ticket_id": {"$ne": exclude_ticket_id}} if exclude_ticket_id else None
        result = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k or settings.top_k, eligible_count),
            where=filters,
            include=["documents", "metadatas", "distances"],
        )
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            {
                "ticket_id": metadata["ticket_id"],
                "evidence": document,
                "distance": distance,
            }
            for document, metadata, distance in zip(
                documents, metadatas, distances, strict=True
            )
        ]
