"""Central settings: paths, model names, API key. Reads .env.

WHY one module: every other file imports its settings from here, so a model name or
folder is changed in exactly one place and experiments stay comparable.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# src/rag/config.py -> parents[2] is the repository root
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Immutable settings. frozen=True so no module can change them by accident."""

    # repr=False: printing a Settings object must never reveal the key.
    groq_api_key: str = field(repr=False)
    groq_base_url: str
    gen_model: str
    judge_model: str
    embedding_model: str
    data_dir: Path = ROOT / "data"
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"
    index_dir: Path = ROOT / "data" / "index"
    cache_dir: Path = ROOT / "data" / "llm_cache"


def load_settings() -> Settings:
    """Load .env, then build Settings. Fails early with a clear message if the key is missing."""
    load_dotenv(ROOT / ".env")
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing: put it in a FILE named .env at the repo root.")
    return Settings(
        groq_api_key=key,
        groq_base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        gen_model=os.environ.get("GEN_MODEL", "qwen/qwen3.8-27b"),
        judge_model=os.environ.get("JUDGE_MODEL", "openai/gpt-oss-120b"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
    )


if __name__ == "__main__":
    s = load_settings()
    print(s)
    print(f"GROQ_API_KEY: set ({len(s.groq_api_key)} chars)")  # length only, never the value
