"""Interfaz local Streamlit para el asistente RAG cardiovascular."""

from __future__ import annotations

from typing import Any

import streamlit as st

from config import COLLECTION_NAME, EMBEDDING_MODEL, VECTOR_SIZE, load_settings
from rag import RAGService


st.set_page_config(
    page_title="CardioDocs | Asistente RAG",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
      --navy: #10243a;
      --blue: #1769aa;
      --blue-soft: #eaf3fb;
      --ink: #1e2937;
      --muted: #64748b;
      --line: #dce4ec;
      --surface: #ffffff;
      --canvas: #f5f7fa;
    }
    .stApp { background: var(--canvas); color: var(--ink); }
    [data-testid="stSidebar"] { background: #10243a; }
    [data-testid="stSidebar"] * { color: #eef5fb; }
    [data-testid="stSidebar"] [data-testid="stMetric"] {
      background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.12);
      padding: .65rem .8rem; border-radius: .65rem;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { color: white; }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #c5d4e3; }
    .brand-kicker { color: #a9c8e4; text-transform: uppercase; letter-spacing: .12em;
      font-size: .72rem; font-weight: 700; margin-bottom: .25rem; }
    .brand-title { color: white; font-size: 1.35rem; line-height: 1.2; font-weight: 750; }
    .brand-description { color: #c5d4e3; font-size: .88rem; line-height: 1.45; margin-top: .55rem; }
    .connection { display: flex; gap: .5rem; align-items: center; color: #d8f4e3;
      background: rgba(34,197,94,.12); border: 1px solid rgba(134,239,172,.25);
      padding: .5rem .65rem; border-radius: .55rem; margin: .9rem 0 1.1rem; font-size: .85rem; }
    .connection-dot { width: .52rem; height: .52rem; background: #4ade80; border-radius: 50%; }
    .page-eyebrow { color: var(--blue); text-transform: uppercase; letter-spacing: .1em;
      font-size: .74rem; font-weight: 750; margin-bottom: .3rem; }
    .welcome-panel { background: linear-gradient(120deg,#fff 0%,#edf5fc 100%);
      border: 1px solid var(--line); border-radius: 1rem; padding: 1.35rem 1.5rem; margin: .5rem 0 1rem; }
    .welcome-panel h2 { color: var(--navy); margin: 0 0 .35rem; font-size: 1.55rem; }
    .welcome-panel p { color: #526477; margin: 0; line-height: 1.55; }
    .section-note { color: var(--muted); font-size: .9rem; }
    .result-summary { background: white; border: 1px solid var(--line); border-radius: .8rem;
      padding: .85rem 1rem; margin: .8rem 0 1rem; }
    div[data-testid="stChatMessage"] { border: 1px solid var(--line); border-radius: .9rem; padding: .9rem 1rem; }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { background: #edf5fc; }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) { background: white; }
    .stButton > button { border-radius: .6rem; }
    .stTabs [data-baseweb="tab-list"] { gap: .4rem; }
    .stTabs [data-baseweb="tab"] { border-radius: .55rem .55rem 0 0; }
    @media (max-width: 700px) {
      .welcome-panel { padding: 1rem; }
      .brand-title { font-size: 1.18rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Carga E5 una sola vez por proceso de Streamlit."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    get_resource_state()["ready"].add("embeddings")
    return model


@st.cache_resource(show_spinner=False)
def get_qdrant_client():
    """Mantiene una conexión Qdrant compartida en el proceso."""
    from qdrant_client import QdrantClient

    settings = load_settings()
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=60,
    )
    get_resource_state()["ready"].add("qdrant")
    return client


@st.cache_resource(show_spinner=False)
def get_gemini_client():
    """Mantiene un cliente Gemini compartido en el proceso."""
    from google import genai
    from google.genai import types

    settings = load_settings()
    client = genai.Client(
        api_key=settings.gemini_api_key,
        http_options=types.HttpOptions(timeout=120000),
    )
    get_resource_state()["ready"].add("gemini")
    return client


@st.cache_resource(show_spinner=False)
def get_resource_state():
    """Registro del proceso para saber qué recursos ya están inicializados."""
    return {"ready": set()}


def get_service(progress_bar=None):
    """Construye el servicio RAG a partir de E5 y clientes cacheados."""
    from google.genai import types

    settings = load_settings()
    ready = get_resource_state()["ready"]

    if progress_bar is not None and "qdrant" not in ready:
        progress_bar.progress(8, text="Conectando con Qdrant…")
    qdrant_client = get_qdrant_client()
    if progress_bar is not None:
        progress_bar.progress(30, text="Conexión con Qdrant preparada")

    if progress_bar is not None and "gemini" not in ready:
        progress_bar.progress(35, text="Inicializando cliente Gemini…")
    gemini_client = get_gemini_client()
    if progress_bar is not None:
        progress_bar.progress(48, text="Cliente Gemini preparado")

    if progress_bar is not None and "embeddings" not in ready:
        progress_bar.progress(52, text="Cargando E5 · intfloat/multilingual-e5-small…")
    embedding_model = get_embedding_model()
    if progress_bar is not None:
        progress_bar.progress(88, text="Modelo E5 listo; verificando la colección…")

    service = RAGService(
        qdrant_client=qdrant_client,
        embedding_model=embedding_model,
        gemini_client=gemini_client,
        gemini_model=settings.gemini_model,
        min_score=settings.rag_min_score,
        collection_name=COLLECTION_NAME,
        generate_config_factory=types.GenerateContentConfig,
    )
    info = service.qdrant.get_collection(COLLECTION_NAME)
    if info.config.params.vectors.size != VECTOR_SIZE:
        raise ValueError("La colección no tiene la dimensión E5 esperada (384).")
    if progress_bar is not None:
        progress_bar.progress(100, text="Sistema listo")
    return settings, service, info


def render_fragment(fragment: dict[str, Any], *, show_source_type: bool = True) -> None:
    """Muestra únicamente campos recibidos desde Qdrant."""
    document = fragment.get("documento") or fragment.get("registro_id") or "Fuente sin nombre"
    page = fragment.get("pagina")
    st.markdown(f"**Documento o fuente:** {document}")
    if page is not None:
        st.markdown(f"**Página:** {page}")
    if fragment.get("chunk_id"):
        st.markdown(f"**Identificador de fragmento:** `{fragment['chunk_id']}`")
    if fragment.get("registro_id"):
        st.markdown(f"**Identificador de registro:** `{fragment['registro_id']}`")
    if show_source_type and fragment.get("tipo_fuente"):
        st.markdown(f"**Tipo de fuente:** `{fragment['tipo_fuente']}`")
    st.markdown(f"**Similitud:** {fragment.get('score', 0):.4f}")
    st.markdown("**Fragmento original**")
    st.markdown(f"> {fragment.get('texto') or 'El punto no contiene texto visible.'}")


def matching_fragment(source: dict[str, Any], fragments: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Asocia la cita con el resultado recuperado sin alterar el contrato RAG."""
    for fragment in fragments:
        if (
            (fragment.get("documento") or fragment.get("registro_id")) == source.get("documento")
            and fragment.get("pagina") == source.get("pagina")
            and fragment.get("score") == source.get("score")
        ):
            return fragment
    return None


def render_assistant_details(message: dict[str, Any]) -> None:
    """Distingue fuentes citadas por Gemini de todos los resultados recuperados."""
    cited = message.get("sources", [])
    recovered = message.get("recovered", [])
    detected = message.get("citations_detected", [])
    invalid = message.get("invalid_citations", [])

    if message.get("score_max") is not None:
        st.caption(f"Mejor similitud recuperada: {message['score_max']:.4f}")

    if message.get("status") == "abstencion":
        st.info("El umbral experimental no encontró evidencia suficiente. Gemini no generó una respuesta.")
    elif not detected or not cited:
        st.warning("La respuesta no contiene citas identificables que correspondan a fuentes recuperadas.")
    if invalid:
        st.warning("Identificadores de cita no encontrados en el contexto: " + ", ".join(invalid))

    if cited:
        with st.expander(f"Fuentes citadas por Gemini ({len(cited)})", expanded=False):
            for source in cited:
                page_text = f" · página {source['pagina']}" if source.get("pagina") else ""
                st.markdown(
                    f"**[{source['id']}] {source.get('documento') or 'Fuente sin nombre'}**"
                    f"{page_text} · similitud {source.get('score', 0):.4f}"
                )
                fragment = matching_fragment(source, recovered)
                if fragment:
                    with st.container(border=True):
                        st.markdown("**Fragmento que sustenta esta cita**")
                        st.markdown(f"> {fragment.get('texto') or 'Sin texto disponible.'}")

    with st.expander(f"Todos los fragmentos recuperados ({len(recovered)})", expanded=False):
        if not recovered:
            st.caption("No se recuperaron fragmentos.")
        for fragment in recovered:
            document = fragment.get("documento") or fragment.get("registro_id") or "Fuente sin nombre"
            page = f" · página {fragment['pagina']}" if fragment.get("pagina") is not None else ""
            with st.container(border=True):
                st.markdown(
                    f"**#{fragment.get('ranking', '—')} · {document}{page}**"
                    f" · similitud {fragment.get('score', 0):.4f}"
                )
                st.caption(f"Tipo de fuente: {fragment.get('tipo_fuente', 'No disponible')}")
                st.markdown(f"> {fragment.get('texto') or 'Sin texto disponible.'}")


def render_message(message: dict[str, Any]) -> None:
    role = message.get("role", "assistant")
    with st.chat_message(role, avatar="👤" if role == "user" else "📄"):
        st.markdown(message.get("content", ""))
        if role == "assistant" and not message.get("error"):
            render_assistant_details(message)
        elif message.get("error"):
            st.caption("Comprueba la conexión local con Qdrant y Gemini e inténtalo de nuevo.")


st.markdown('<div class="page-eyebrow">Base de conocimiento cardiovascular</div>', unsafe_allow_html=True)
st.title("Asistente documental cardiovascular")
st.caption("Consulta documentos recuperados, contrasta fragmentos y sigue las citas hasta su fuente.")

ready_resources = get_resource_state()["ready"]
startup_progress = None
startup_placeholder = None
if not {"qdrant", "gemini", "embeddings"}.issubset(ready_resources):
    startup_placeholder = st.empty()
    with startup_placeholder.container():
        st.markdown("#### Preparando CardioDocs")
        st.caption("La primera carga puede tardar mientras se conecta y carga el modelo local de embeddings.")
        startup_progress = st.progress(0, text="Validando configuración local…")

try:
    settings, service, collection_info = get_service(progress_bar=startup_progress)
    if startup_placeholder is not None:
        startup_placeholder.empty()
except Exception as exc:
    st.error("No se pudo iniciar la aplicación. Revisa la configuración local y la conectividad.")
    st.caption(f"Detalle técnico: {type(exc).__name__}")
    st.stop()


with st.sidebar:
    st.markdown('<div class="brand-kicker">Inteligencia documental en salud</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-title">CardioDocs RAG</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-description">Asistente académico para explorar evidencia cardiovascular y revisar sus fuentes.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="connection"><span class="connection-dot"></span><span>Qdrant conectado</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Colección**")
    st.markdown(f"**{COLLECTION_NAME}**")
    point_count = collection_info.points_count
    st.metric("Puntos vectoriales", f"{point_count:,}" if point_count is not None else "No disponible")
    st.markdown("**Modelo Gemini**")
    st.markdown(settings.gemini_model)
    st.markdown("**Umbral de relevancia**")
    st.markdown(f"**{settings.rag_min_score:.2f}** · experimental")
    top_k = st.slider("Fragmentos por consulta", min_value=1, max_value=10, value=5)
    st.caption("Conexión de consulta. Esta interfaz no modifica los puntos de Qdrant.")


chat_tab, explore_tab = st.tabs(["Chat documental", "Explorador documental"])

if "messages" not in st.session_state:
    st.session_state.messages = []

with chat_tab:
    chat_header, clear_col = st.columns([5, 1])
    with chat_header:
        st.subheader("Conversación")
    with clear_col:
        if st.session_state.messages and st.button("Limpiar", use_container_width=True, key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

    if not st.session_state.messages:
        st.markdown(
            '<div class="welcome-panel"><h2>Pregúntale al corpus</h2>'
            '<p>Formula una pregunta sobre salud cardiovascular. El asistente buscará fragmentos en la colección y responderá con evidencia recuperada. Si la evidencia no supera el umbral configurado, se abstendrá.</p></div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Preguntas para iniciar la demostración**")
        suggestions = [
            "¿Cuáles son los principales factores de riesgo cardiovascular?",
            "¿Cómo se relacionan la hipertensión y la diabetes con el riesgo cardiovascular?",
            "¿Qué antibiótico debo tomar para una infección urinaria?",
        ]
        suggestion_cols = st.columns(len(suggestions))
        selected_suggestion = None
        for index, (column, suggestion) in enumerate(zip(suggestion_cols, suggestions)):
            with column:
                if st.button(suggestion, key=f"suggestion_{index}", use_container_width=True):
                    selected_suggestion = suggestion
    else:
        selected_suggestion = None

    for message in st.session_state.messages:
        render_message(message)

    typed_question = st.chat_input("Pregunta sobre documentos cardiovasculares…")
    question = selected_suggestion or typed_question
    if question:
        prior_history = [
            {"role": item["role"], "content": item["content"]}
            for item in st.session_state.messages[-6:]
            if item.get("role") in {"user", "assistant"}
        ]
        st.session_state.messages.append({"role": "user", "content": question})
        try:
            with st.spinner("Buscando evidencia documental y preparando la respuesta…"):
                result = service.answer(question, top_k=top_k, history=prior_history)
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.get("respuesta", "No se recibió una respuesta."),
                "sources": result.get("fuentes", []),
                "recovered": result.get("fragmentos", []),
                "citations_detected": result.get("citas_detectadas", []),
                "invalid_citations": result.get("citas_invalidas", []),
                "score_max": result.get("score_maximo"),
                "status": result.get("estado"),
            })
        except Exception as exc:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "No pude completar la consulta. Revisa la conexión con los servicios configurados e inténtalo de nuevo.",
                "error": True,
                "error_type": type(exc).__name__,
            })
        st.rerun()


with explore_tab:
    st.subheader("Explorar fragmentos recuperados")
    st.markdown(
        '<p class="section-note">Busca por significado y revisa qué devuelve la colección para una consulta. Los filtros de documento se aplican a los resultados del Top‑K mostrado.</p>',
        unsafe_allow_html=True,
    )
    with st.form("document_search_form"):
        explore_query = st.text_input(
            "Consulta semántica",
            placeholder="Ej.: factores de riesgo cardiovascular",
            key="explore_query_input",
        )
        source_label = st.selectbox(
            "Tipo de fuente",
            ["Todas las fuentes", "Documentos PDF", "Dataset estructurado"],
            key="explore_source_type",
        )
        search_submitted = st.form_submit_button("Buscar en la colección", type="primary")

    if search_submitted:
        if not explore_query.strip():
            st.warning("Escribe una consulta para iniciar la búsqueda semántica.")
        else:
            source_value = {
                "Todas las fuentes": None,
                "Documentos PDF": "documento_pdf",
                "Dataset estructurado": "dataset_estructurado",
            }[source_label]
            try:
                with st.spinner("Consultando Qdrant…"):
                    found = service.search(explore_query, top_k=top_k, source_type=source_value)
                st.session_state.explorer_results = found
                st.session_state.explorer_query = explore_query
                st.session_state.explorer_error = None
            except Exception as exc:
                st.session_state.explorer_results = []
                st.session_state.explorer_query = explore_query
                st.session_state.explorer_error = type(exc).__name__

    if st.session_state.get("explorer_error"):
        st.error("Falló la búsqueda. Revisa la conexión con Qdrant e inténtalo de nuevo.")
        st.caption(f"Detalle técnico: {st.session_state.explorer_error}")

    results = st.session_state.get("explorer_results")
    if results is not None and not st.session_state.get("explorer_error"):
        st.markdown(f"**Consulta:** {st.session_state.get('explorer_query', '')}")
        if not results:
            st.info("La búsqueda no devolvió fragmentos para esta consulta.")
        else:
            document_names = sorted({
                str(item.get("documento") or item.get("registro_id") or "Fuente sin nombre")
                for item in results
            })
            filter_col, _ = st.columns([2, 4])
            with filter_col:
                selected_document = st.selectbox(
                    "Filtrar por documento o fuente",
                    ["Todas las fuentes"] + document_names,
                    key="explorer_document_filter",
                )
            visible_results = results if selected_document == "Todas las fuentes" else [
                item for item in results
                if str(item.get("documento") or item.get("registro_id") or "Fuente sin nombre") == selected_document
            ]
            distinct_sources = len(document_names)
            best_score = max(item["score"] for item in results)
            metric_cols = st.columns(3)
            metric_cols[0].metric("Fragmentos recuperados", len(results))
            metric_cols[1].metric("Fuentes distintas en el Top‑K", distinct_sources)
            metric_cols[2].metric("Mejor similitud", f"{best_score:.4f}")
            st.caption(f"Mostrando {len(visible_results)} de {len(results)} fragmentos recuperados.")
            for fragment in visible_results:
                document = fragment.get("documento") or fragment.get("registro_id") or "Fuente sin nombre"
                page_label = f" · página {fragment['pagina']}" if fragment.get("pagina") is not None else ""
                chunk_label = f" · {fragment['chunk_id']}" if fragment.get("chunk_id") else ""
                with st.expander(
                    f"#{fragment['ranking']} · {document}{page_label}{chunk_label} · similitud {fragment['score']:.4f}"
                ):
                    render_fragment(fragment)
