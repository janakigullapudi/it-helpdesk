from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    model: str
    embedding_model: str
    data_dir: Path
    chroma_dir: Path
    session_dir: Path
    top_k: int


def _project_path(variable: str, default: str) -> Path:
    path = Path(os.getenv(variable, default))
    return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    top_k = int(os.getenv("HELPDESK_TOP_K", "5"))
    if not 1 <= top_k <= 20:
        raise ValueError("HELPDESK_TOP_K must be between 1 and 20.")

    return Settings(
        api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        embedding_model=os.getenv(
            "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"
        ),
        data_dir=_project_path("HELPDESK_DATA_DIR", "data"        ),
        chroma_dir=_project_path("HELPDESK_CHROMA_DIR", ".chroma"),
        session_dir=_project_path("HELPDESK_SESSION_DIR", "sessions"),
        top_k=top_k,
    )


def require_api_key() -> str:
    key = get_settings().api_key
    if not key:
        raise RuntimeError(
            "Gemini API key is missing. Set GEMINI_API_KEY in your environment "
            "or copy .env.example to .env and add your key."
        )
    return key
