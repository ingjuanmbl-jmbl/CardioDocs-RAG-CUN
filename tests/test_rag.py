from types import SimpleNamespace
from unittest.mock import Mock

import pytest

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
