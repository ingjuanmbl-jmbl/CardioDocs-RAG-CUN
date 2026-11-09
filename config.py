"""Configuración local de la aplicación ACA2 RAG."""

from dataclasses import dataclass
import os

from dotenv import load_dotenv


COLLECTION_NAME = "corpus_cardiovascular"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
VECTOR_SIZE = 384


@dataclass(frozen=True)
class Settings:
    qdrant_url: str
    qdrant_api_key: str
    gemini_api_key: str
    gemini_model: str
    rag_min_score: float


def load_settings() -> Settings:
    """Lee secretos desde .env y valida el umbral de similitud."""
    load_dotenv()
    required = {
        "QDRANT_URL": os.getenv("QDRANT_URL", "").strip(),
        "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY", "").strip(),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", "").strip(),
        "GEMINI_MODEL": os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip(),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            "Faltan variables de configuración: " + ", ".join(missing)
            + ". Configúralas en el archivo .env a partir de .env.example."
        )

    raw_score = os.getenv("RAG_MIN_SCORE", "0.88").strip()
    try:
        score = float(raw_score)
    except ValueError as exc:
        raise ValueError("RAG_MIN_SCORE debe ser un número entre 0 y 1.") from exc
    if not 0 <= score <= 1:
        raise ValueError("RAG_MIN_SCORE debe estar entre 0 y 1.")

    return Settings(
        qdrant_url=required["QDRANT_URL"],
        qdrant_api_key=required["QDRANT_API_KEY"],
        gemini_api_key=required["GEMINI_API_KEY"],
        gemini_model=required["GEMINI_MODEL"],
        rag_min_score=score,
    )
