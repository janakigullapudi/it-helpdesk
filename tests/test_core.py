from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from helpdesk.auth import authenticate_user, find_mock_user
from helpdesk.knowledge import search_knowledge
from helpdesk.evaluation import token_f1
from helpdesk.privacy import redact_pii
from helpdesk.registry import ToolRegistry, list_mock_tickets
from helpdesk.settings import PROJECT_ROOT, get_settings


class _FakeEmbeddings:
    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [[1.0, float(index + 1)] for index in range(len(documents))]

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 1.0]


class _FakeCollection:
    def __init__(self) -> None:
        self.ids: list[str] = []
        self.added_documents: list[str] = []
        self.added_embeddings: list[list[float]] = []
        self.query_arguments: dict[str, object] = {}

    def get(self, *, include: list[str]) -> dict[str, list[str]]:
        return {"ids": self.ids}

    def add(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, str]],
        embeddings: list[list[float]],
    ) -> None:
        self.ids = ids
        self.added_documents = documents
        self.added_embeddings = embeddings

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, object],
        include: list[str],
    ) -> dict[str, list[list[object]]]:
        self.query_arguments = {"n_results": n_results, "where": where}
        ids = [ticket_id for ticket_id in self.ids if ticket_id != where["ticket_id"]["$ne"]]
        selected = ids[:n_results]
        return {
            "documents": [[f"Evidence for {ticket_id}" for ticket_id in selected]],
            "metadatas": [[{"ticket_id": ticket_id} for ticket_id in selected]],
            "distances": [[0.1 for _ in selected]],
        }


class CoreTests(unittest.TestCase):
    def test_approved_mock_ticket_is_persisted_to_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = SimpleNamespace(data_dir=Path(directory))
            registry = ToolRegistry(PROJECT_ROOT / "config" / "tools.yaml")
            with patch("helpdesk.registry.get_settings", return_value=settings):
                result = registry.execute(
                    "create_mock_ticket",
                    {
                        "query": "VPN disconnects",
                        "summary": "Investigate VPN connectivity",
                        "user_id": "E1001",
                    },
                    approved=True,
                )

            database_path = Path(directory) / "mock_tickets.sqlite"
            with patch("helpdesk.registry.get_settings", return_value=settings):
                tickets = list_mock_tickets()
            self.assertTrue(database_path.exists())

        self.assertEqual(len(tickets), 1)
        self.assertEqual(
            tuple(
                tickets[0][key]
                for key in ("ticket_id", "user_id", "summary", "query", "status")
            ),
            (
                result["ticket_id"],
                "E1001",
                "Investigate VPN connectivity",
                "VPN disconnects",
                "created",
            ),
        )
        self.assertTrue(tickets[0]["created_at"])

    def test_mock_login_accepts_demo_user_and_rejects_bad_credentials(self) -> None:
        user = authenticate_user("e1001", "AlexDemo!2026")
        self.assertEqual(
            user,
            {
                "user_id": "E1001",
                "display_name": "Alex Chen",
                "department": "IT",
            },
        )
        self.assertIsNone(authenticate_user("E1001", "wrong password"))
        self.assertIsNone(authenticate_user("unknown", "wrong password"))

    def test_access_check_verifies_known_mock_user_only(self) -> None:
        registry = ToolRegistry(PROJECT_ROOT / "config" / "tools.yaml")
        verified = registry.execute(
            "check_access",
            {"query": "Cannot log in", "user_id": "E1001"},
        )
        unverified = registry.execute(
            "check_access",
            {"query": "Cannot log in"},
        )
        self.assertTrue(verified["identity_verified"])
        self.assertFalse(verified["access_changed"])
        self.assertIn("verified", verified["message"])
        self.assertFalse(unverified["identity_verified"])
        self.assertIsNone(find_mock_user("not-a-user"))

    def test_token_f1_exact_and_empty(self) -> None:
        self.assertEqual(token_f1("Reset account credentials", "Reset account credentials"), 1.0)
        self.assertEqual(token_f1("", "Reset account credentials"), 0.0)

    def test_token_f1_partial_overlap(self) -> None:
        self.assertAlmostEqual(token_f1("reset credentials", "reset account credentials"), 0.8)

    def test_redacts_email_and_phone(self) -> None:
        result = redact_pii("Contact jane@example.com at +1 (555) 123-4567")
        self.assertNotIn("jane@example.com", result)
        self.assertNotIn("555", result)

    def test_registry_enforces_approval(self) -> None:
        registry = ToolRegistry(PROJECT_ROOT / "config" / "tools.yaml")
        with self.assertRaises(PermissionError):
            registry.execute(
                "create_mock_ticket",
                {"query": "Need help", "summary": "Investigate login"},
            )

    def test_access_check_is_non_mutating(self) -> None:
        registry = ToolRegistry(PROJECT_ROOT / "config" / "tools.yaml")
        result = registry.execute("check_access", {"query": "Cannot log in"})
        self.assertFalse(result["identity_verified"])
        self.assertFalse(result["access_changed"])

    def test_chroma_indexes_and_excludes_evaluation_ticket(self) -> None:
        records = json.loads((get_settings().data_dir / "golden_subset.json").read_text(encoding="utf-8"))
        excluded_id = records[0]["ticket_id"]
        collection = _FakeCollection()
        client = type(
            "FakeClient",
            (),
            {"get_or_create_collection": lambda self, *args, **kwargs: collection},
        )()
        with (
            patch("helpdesk.knowledge.PersistentClient", return_value=client),
            patch("helpdesk.knowledge._embeddings", return_value=_FakeEmbeddings()),
        ):
            hits = search_knowledge(
                "Cannot sign in", exclude_ticket_id=excluded_id, top_k=5
            )
        self.assertEqual(len(hits), 5)
        self.assertNotIn(excluded_id, {hit["ticket_id"] for hit in hits})
        self.assertEqual(len(collection.added_documents), 50)
        self.assertEqual(len(collection.added_embeddings), 50)
        self.assertEqual(collection.query_arguments["n_results"], 5)
        self.assertEqual(
            collection.query_arguments["where"],
            {"ticket_id": {"$ne": excluded_id}},
        )

    def test_chroma_native_load_failure_has_windows_guidance(self) -> None:
        with patch(
            "helpdesk.knowledge.PersistentClient",
            side_effect=ImportError("chromadb_rust_bindings not found"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Visual C\\+\\+"):
                search_knowledge("Cannot sign in")


if __name__ == "__main__":
    unittest.main()
