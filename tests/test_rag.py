from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import config
from config import Settings, load_settings
from rag import ABSTENTION_TEXT, RAGService


def make_service(score=0.93, text="Contenido cardiovascular de prueba.", answer="Respuesta basada en [F1]."):
    point = SimpleNamespace(
        score=score,
        payload={
            "tipo_fuente": "documento_pdf",
            "documento": "guia.pdf",
            "pagina": 7,
            "chunk_id": "chunk_000001",
            "texto": text,
        },
    )
    qdrant = Mock()
    qdrant.query_points.return_value = SimpleNamespace(points=[point])
    embedding = Mock()
    embedding.encode.return_value = [[0.1, 0.2, 0.3]]
    gemini = Mock()
    gemini.models.generate_content.return_value = SimpleNamespace(text=answer)
    service = RAGService(qdrant, embedding, gemini, "mock-model", min_score=0.88)
    return service, qdrant, embedding, gemini


def test_search_uses_query_prefix_and_returns_citation_metadata():
    service, qdrant, embedding, _ = make_service()

    results = service.search("¿Qué es el riesgo cardiovascular?", top_k=3)

    assert embedding.encode.call_args.args[0] == ["query: ¿Qué es el riesgo cardiovascular?"]
    qdrant.query_points.assert_called_once()
    assert qdrant.query_points.call_args.kwargs["collection_name"] == "corpus_cardiovascular"
    assert qdrant.query_points.call_args.kwargs["limit"] == 3
    assert results[0]["documento"] == "guia.pdf"
    assert results[0]["pagina"] == 7
    assert results[0]["score"] == 0.93


def test_answer_abstains_below_threshold_without_calling_gemini():
    service, _, _, gemini = make_service(score=0.72)

    result = service.answer("Pregunta fuera del corpus")

    assert result["estado"] == "abstencion"
    assert result["respuesta"] == ABSTENTION_TEXT
    assert result["score_maximo"] == 0.72
    gemini.models.generate_content.assert_not_called()


def test_answer_generates_response_and_maps_valid_citations():
    service, _, _, gemini = make_service(answer="El documento describe el factor [F1].")

    result = service.answer("Pregunta respaldada")

    assert result["estado"] == "respuesta_generada"
    assert result["fuentes"] == [{
        "id": "F1", "documento": "guia.pdf", "pagina": 7, "score": 0.93
    }]
    assert result["citas_invalidas"] == []
    assert gemini.models.generate_content.call_args.kwargs["model"] == "mock-model"


def test_answer_warns_for_unknown_citation_and_includes_history():
    service, _, _, gemini = make_service(answer="Hallazgo [F8].")

    result = service.answer(
        "¿Y qué implica?",
        history=[{"role": "user", "content": "Pregunta previa"}],
    )

    prompt = gemini.models.generate_content.call_args.kwargs["contents"]
    assert "Usuario: Pregunta previa" in prompt
    assert result["citas_invalidas"] == ["F8"]
    assert "F8" in result["respuesta"]


def test_empty_question_does_not_call_qdrant():
    service, qdrant, _, _ = make_service()

    assert service.search("   ") == []
    qdrant.query_points.assert_not_called()


def test_settings_require_credentials_and_validate_score(monkeypatch):
    monkeypatch.setattr(config, "_streamlit_secret", lambda name: None)
    values = {
        "QDRANT_URL": "https://example.qdrant.io",
        "QDRANT_API_KEY": "fake-qdrant-key",
        "GEMINI_API_KEY": "fake-gemini-key",
        "GEMINI_MODEL": "gemini-3.5-flash-lite",
        "RAG_MIN_SCORE": "0.88",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    settings = load_settings()
    assert isinstance(settings, Settings)
    assert settings.rag_min_score == 0.88

    monkeypatch.setenv("RAG_MIN_SCORE", "1.2")
    with pytest.raises(ValueError, match="entre 0 y 1"):
        load_settings()


def test_settings_prioritize_streamlit_secrets_over_environment(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", Mock())
    secrets = {
        "QDRANT_URL": "https://cloud.example.qdrant.io",
        "QDRANT_API_KEY": "fake-cloud-qdrant-key",
        "GEMINI_API_KEY": "fake-cloud-gemini-key",
        "GEMINI_MODEL": "gemini-cloud-mock",
        "RAG_MIN_SCORE": "0.91",
    }
    monkeypatch.setattr(config, "_streamlit_secret", secrets.get)
    for name in secrets:
        monkeypatch.setenv(name, "local-conflicting-value")

    settings = load_settings()

    assert settings.qdrant_url == secrets["QDRANT_URL"]
    assert settings.qdrant_api_key == secrets["QDRANT_API_KEY"]
    assert settings.gemini_api_key == secrets["GEMINI_API_KEY"]
    assert settings.gemini_model == secrets["GEMINI_MODEL"]
    assert settings.rag_min_score == 0.91


def test_settings_use_dotenv_values_when_streamlit_secrets_are_unavailable(monkeypatch):
    values = {
        "QDRANT_URL": "https://local.example.qdrant.io",
        "QDRANT_API_KEY": "fake-local-qdrant-key",
        "GEMINI_API_KEY": "fake-local-gemini-key",
        "GEMINI_MODEL": "gemini-local-mock",
        "RAG_MIN_SCORE": "0.87",
    }
    for name in values:
        monkeypatch.delenv(name, raising=False)

    def fake_load_dotenv():
        for name, value in values.items():
            monkeypatch.setenv(name, value)

    monkeypatch.setattr(config, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(config, "_streamlit_secret", lambda name: None)

    settings = load_settings()

    assert settings.qdrant_url == values["QDRANT_URL"]
    assert settings.qdrant_api_key == values["QDRANT_API_KEY"]
    assert settings.gemini_api_key == values["GEMINI_API_KEY"]
    assert settings.gemini_model == values["GEMINI_MODEL"]
    assert settings.rag_min_score == 0.87


def test_settings_report_missing_secret_names_without_values(monkeypatch):
    for name in ("QDRANT_URL", "QDRANT_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "load_dotenv", Mock())
    monkeypatch.setattr(config, "_streamlit_secret", lambda name: None)

    with pytest.raises(ValueError) as error:
        load_settings()

    message = str(error.value)
    assert "QDRANT_URL" in message
    assert "QDRANT_API_KEY" in message
    assert "GEMINI_API_KEY" in message
    assert "fake-local" not in message
