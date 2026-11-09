"""Configuración local y de Streamlit Community Cloud para ACA2 RAG."""

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
    """Prioriza st.secrets y usa variables de entorno/.env como alternativa.

    Los valores de credenciales nunca se incluyen en mensajes ni se imprimen.
    """
    load_dotenv()
    values = {
        "QDRANT_URL": _setting("QDRANT_URL"),
        "QDRANT_API_KEY": _setting("QDRANT_API_KEY"),
        "GEMINI_API_KEY": _setting("GEMINI_API_KEY"),
        "GEMINI_MODEL": _setting("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    }
    missing = [
        name for name in ("QDRANT_URL", "QDRANT_API_KEY", "GEMINI_API_KEY")
        if not values[name]
    ]
    if missing:
        raise ValueError(
            "Faltan variables de configuración: " + ", ".join(missing)
            + ". Configúralas en st.secrets o en el archivo .env local."
        )

    raw_score = _setting("RAG_MIN_SCORE", "0.88")
    try:
        score = float(raw_score)
    except ValueError as exc:
        raise ValueError("RAG_MIN_SCORE debe ser un número entre 0 y 1.") from exc
    if not 0 <= score <= 1:
        raise ValueError("RAG_MIN_SCORE debe estar entre 0 y 1.")

    return Settings(
        qdrant_url=values["QDRANT_URL"],
        qdrant_api_key=values["QDRANT_API_KEY"],
        gemini_api_key=values["GEMINI_API_KEY"],
        gemini_model=values["GEMINI_MODEL"],
        rag_min_score=score,
    )


def _setting(name: str, default: str = "") -> str:
    """Obtiene una clave de Streamlit y, si falta, de os.environ/.env."""
    secret_value = _streamlit_secret(name)
    if secret_value is not None:
        normalized = str(secret_value).strip()
        if normalized:
            return normalized
    return os.getenv(name, default).strip()


def _streamlit_secret(name: str):
    """Lee st.secrets sin romper la ejecución local sin secrets.toml."""
    try:
        import streamlit as st

        value = st.secrets.get(name)
    except Exception:
        # Streamlit lanza una excepción cuando no hay secrets configurados o
        # cuando config.py se ejecuta fuera de un contexto de Streamlit.
        return None
    return value
