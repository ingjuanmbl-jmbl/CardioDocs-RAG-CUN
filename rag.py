"""Recuperación y generación RAG basada en el notebook ACA2_RAG."""

from __future__ import annotations

import re
from typing import Any, Iterable

from config import COLLECTION_NAME, EMBEDDING_MODEL


ABSTENTION_TEXT = (
    "No encontré evidencia documental suficientemente pertinente en la "
    "base de conocimiento cardiovascular para responder esta pregunta. "
    "Por tanto, no generaré una respuesta sin respaldo documental."
)


class RAGService:
    """Servicio RAG con dependencias inyectables para pruebas con mocks."""

    def __init__(
        self,
        qdrant_client: Any,
        embedding_model: Any,
        gemini_client: Any,
        gemini_model: str,
        min_score: float = 0.88,
        collection_name: str = COLLECTION_NAME,
        generate_config_factory: Any = None,
    ) -> None:
        if not 0 <= min_score <= 1:
            raise ValueError("min_score debe estar entre 0 y 1.")
        self.qdrant = qdrant_client
        self.embedding_model = embedding_model
        self.gemini = gemini_client
        self.gemini_model = gemini_model
        self.min_score = min_score
        self.collection_name = collection_name
        self.generate_config_factory = generate_config_factory

    def search(
        self, question: str, top_k: int = 5, source_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Recupera fragmentos usando el prefijo query: verificado en ACA2."""
        question = question.strip()
        if not question:
            return []
        if top_k < 1:
            raise ValueError("top_k debe ser mayor que cero.")

        vector = self.embedding_model.encode(
            ["query: " + question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        vector = vector.tolist() if hasattr(vector, "tolist") else list(vector)

        query_filter = None
        if source_type:
            from qdrant_client import models

            query_filter = models.Filter(
                must=[models.FieldCondition(
                    key="tipo_fuente",
                    match=models.MatchValue(value=source_type),
                )]
            )

        response = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        output = []
        for rank, point in enumerate(response.points, start=1):
            payload = point.payload or {}
            output.append({
                "ranking": rank,
                "score": round(float(point.score), 4),
                "tipo_fuente": payload.get("tipo_fuente", "sin_clasificar"),
                "documento": payload.get("documento") or payload.get("dataset"),
                "pagina": payload.get("pagina"),
                "registro_id": payload.get("registro_id"),
                "chunk_id": payload.get("chunk_id"),
                "texto": payload.get("texto", ""),
            })
        return output

    def answer(
        self,
        question: str,
        top_k: int = 5,
        history: Iterable[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Responde con Gemini solo si hay evidencia sobre el umbral."""
        results = self.search(question, top_k=top_k)
        if not results:
            return self._empty_result("sin_resultados", "No se encontraron documentos para consultar.")

        max_score = max(item["score"] for item in results)
        relevant = [
            item for item in results
            if item["score"] >= self.min_score and item.get("texto", "").strip()
        ]
        if not relevant:
            return {
                "estado": "abstencion",
                "respuesta": ABSTENTION_TEXT,
                "fuentes": [],
                "fuentes_contexto": [],
                "fragmentos": results,
                "score_maximo": max_score,
                "citas_detectadas": [],
                "citas_invalidas": [],
            }

        context_parts: list[str] = []
        context_sources: list[dict[str, Any]] = []
        for position, item in enumerate(relevant, start=1):
            source_id = f"F{position}"
            document = item.get("documento") or "Documento no identificado"
            page = item.get("pagina") or "No disponible"
            context_parts.append(
                f"[{source_id}]\nDocumento: {document}\nPágina: {page}\n"
                f"Contenido:\n{item['texto']}"
            )
            context_sources.append({
                "id": source_id,
                "documento": document,
                "pagina": page,
                "score": item["score"],
            })

        history_text = self._format_history(history)
        prompt = f"""Eres un asistente académico especializado en documentación cardiovascular.

Responde exclusivamente con la evidencia del CONTEXTO DOCUMENTAL. Usa el
historial únicamente para entender referencias conversacionales; no es fuente
de hechos. Responde en español, no inventes información, cita cada afirmación
relevante con identificadores como [F1], no inventes citas ni páginas y
reconoce lo que los documentos no permitan responder. No realices diagnósticos
individuales ni recomiendes medicamentos o tratamientos individuales.

HISTORIAL RECIENTE:
{history_text or "Sin historial previo."}

PREGUNTA:
{question}

CONTEXTO DOCUMENTAL:
{chr(10).join(context_parts)}

REDACTA UNA RESPUESTA FUNDAMENTADA:
"""

        kwargs: dict[str, Any] = {
            "model": self.gemini_model,
            "contents": prompt,
        }
        if self.generate_config_factory:
            kwargs["config"] = self.generate_config_factory(
                temperature=0.2, max_output_tokens=800
            )
        else:
            try:
                from google.genai import types
                kwargs["config"] = types.GenerateContentConfig(
                    temperature=0.2, max_output_tokens=800
                )
            except ImportError:
                # Permite pruebas con mocks sin instalar el SDK.
                pass
        response = self.gemini.models.generate_content(**kwargs)
        generated = getattr(response, "text", None)
        if not generated:
            raise ValueError("Gemini no generó una respuesta.")

        detected = list(dict.fromkeys(re.findall(r"\[(F\d+)\]", generated)))
        valid_ids = {source["id"] for source in context_sources}
        invalid = [citation for citation in detected if citation not in valid_ids]
        cited_sources = [s for s in context_sources if s["id"] in detected]
        answer = generated
        if not detected:
            answer += "\n\nAdvertencia: la respuesta no contiene citas documentales identificables."
        if invalid:
            answer += "\n\nAdvertencia: se detectaron citas no presentes en el contexto: " + ", ".join(invalid)

        return {
            "estado": "respuesta_generada",
            "respuesta": answer,
            "fuentes": cited_sources,
            "fuentes_contexto": context_sources,
            "fragmentos": results,
            "score_maximo": max_score,
            "citas_detectadas": detected,
            "citas_invalidas": invalid,
        }

    @staticmethod
    def _format_history(history: Iterable[dict[str, str]] | None) -> str:
        if not history:
            return ""
        lines = []
        for item in list(history)[-6:]:
            role = "Usuario" if item.get("role") == "user" else "Asistente"
            content = item.get("content", "").strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _empty_result(state: str, message: str) -> dict[str, Any]:
        return {
            "estado": state,
            "respuesta": message,
            "fuentes": [],
            "fuentes_contexto": [],
            "fragmentos": [],
            "score_maximo": None,
            "citas_detectadas": [],
            "citas_invalidas": [],
        }


def create_service(settings: Any) -> RAGService:
    """Construye clientes en modo de consulta, sin operaciones de escritura."""
    from google import genai
    from google.genai import types
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    qdrant = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=60,
    )
    gemini = genai.Client(
        api_key=settings.gemini_api_key,
        http_options=types.HttpOptions(timeout=120000),
    )
    embedding = SentenceTransformer(EMBEDDING_MODEL)
    return RAGService(
        qdrant_client=qdrant,
        embedding_model=embedding,
        gemini_client=gemini,
        gemini_model=settings.gemini_model,
        min_score=settings.rag_min_score,
        generate_config_factory=types.GenerateContentConfig,
    )
