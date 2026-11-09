# CardioDocs RAG: asistente documental cardiovascular

Aplicación académica local para explorar y consultar la colección existente `corpus_cardiovascular` de Qdrant Cloud. Recupera fragmentos con E5 y utiliza Gemini para redactar respuestas fundamentadas en ellos. La aplicación funciona en modo de consulta: no crea, actualiza ni elimina puntos en Qdrant.

## Funcionalidades

- Chat en español con historial guardado durante la sesión de Streamlit.
- Preguntas sugeridas para facilitar demostraciones.
- Respuestas en Markdown con identificadores de cita como `[F1]`.
- Panel de fuentes citadas por Gemini y panel separado con todos los fragmentos recuperados; ambos permiten revisar el texto disponible, el documento, la página y el score.
- Advertencias cuando la respuesta no incluye citas identificables o menciona identificadores que no aparecen en el contexto.
- Abstención experimental cuando ningún fragmento con texto alcanza el umbral configurado.
- Explorador con búsqueda semántica, filtro por tipo de fuente y filtro por documento entre los resultados del Top‑K actual.
- Barra de carga por etapas en la primera inicialización del proceso. Desaparece cuando el sistema está listo y no se vuelve a mostrar en los reruns mientras los recursos sigan en caché.
- Tema claro fijado desde `.streamlit/config.toml` para mantener el contraste en Streamlit Community Cloud.

El filtro de documento se aplica a los resultados ya recuperados para esa búsqueda; no amplía el Top‑K ni hace una consulta global adicional por documento.

## Base técnica verificada

Se inspeccionaron los notebooks locales `notebooks/ACA1 (1)` y `notebooks/ACA2_RAG`. Son notebooks JSON sin extensión y permanecieron sin cambios.

- **Colección:** `corpus_cardiovascular`. ACA2 registró 2.152 puntos y vectores de 384 dimensiones al momento de su validación; la aplicación consulta el conteo actual al conectarse.
- **Embeddings:** `intfloat/multilingual-e5-small`, normalizados y con el prefijo `query:` para las consultas.
- **Recuperación:** Qdrant `query_points`, con payload y sin recuperar los vectores.
- **Generación:** SDK `google-genai`, modelo por defecto `gemini-3.5-flash-lite`, temperatura 0,2 y máximo de 800 tokens.
- **Citas:** el contexto se etiqueta con `[F1]`, `[F2]`, etc., junto con documento y página. La app verifica que los identificadores generados existan entre las fuentes del contexto; esto no prueba que cada cita respalde semánticamente la afirmación asociada.
- **Umbral:** `RAG_MIN_SCORE` usa 0,88 por defecto. Es el valor experimental del notebook, no un umbral clínico validado.

El historial reciente se incluye para interpretar referencias conversacionales, pero el prompt indica a Gemini que use los documentos recuperados como única fuente de hechos. La herramienta es académica y no realiza diagnósticos individuales.

## Requisitos e instalación

Usa Python 3.10 o posterior. En PowerShell, desde la carpeta del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

`config.py` busca cada variable primero en `st.secrets` y, si no está definida allí, la toma de `os.environ` o de `.env`. Esta prioridad permite usar la misma configuración en local y en Streamlit Community Cloud.

Para ejecución local, completa `.env` con estas variables:

```text
QDRANT_URL
QDRANT_API_KEY
GEMINI_API_KEY
GEMINI_MODEL
RAG_MIN_SCORE
```

`GEMINI_MODEL` tiene como valor por defecto `gemini-3.5-flash-lite`; `RAG_MIN_SCORE`, `0.88`. `.env.example` contiene valores ficticios. `.env` está excluido por `.gitignore`; no incluyas credenciales reales en el código ni compartas ese archivo.

En Streamlit Community Cloud, define esas mismas cinco claves en **App settings → Secrets**. No es necesario cambiar el código ni subir `.env`; las claves configuradas en `st.secrets` tienen prioridad sobre las variables locales. Si falta `QDRANT_URL`, `QDRANT_API_KEY` o `GEMINI_API_KEY` en ambas fuentes, la aplicación muestra un error con los nombres faltantes, nunca sus valores.

La primera inicialización de E5 puede descargar el modelo a la caché local de Sentence Transformers. La barra de CardioDocs indica las etapas de conexión con Qdrant, preparación de Gemini, carga de E5 y verificación de la colección.

## Caché y recargas

Streamlit conserva el modelo E5 y los clientes de Qdrant y Gemini con `st.cache_resource` durante la vida del proceso. Recargar la página del navegador o provocar un rerun no equivale a reiniciar el servidor: los recursos permanecen en caché y se reutilizan. Al detener Streamlit y volverlo a iniciar, se inicializan de nuevo.

Si cambias una credencial en `.env`, detén el servidor con `Ctrl+C` y vuelve a iniciarlo para que los clientes se creen con los nuevos valores. También puedes limpiar la caché de recursos desde el menú de Streamlit o invocar `st.cache_resource.clear()` desde el código.

## Ejecución

```powershell
streamlit run app.py
```

## Pruebas

Las seis pruebas automatizadas usan mocks para validar recuperación, abstención, citas, historial y configuración sin llamar a Qdrant o Gemini:

```powershell
python -m pytest tests\test_rag.py -q -p no:cacheprovider
```

## Límites de interpretación

Un score alto indica similitud vectorial, no garantiza que el fragmento responda correcta o completamente la pregunta. La validación automática de citas comprueba identificadores; no evalúa si cada cita sustenta semánticamente una afirmación. El umbral y los resultados son experimentales y no constituyen validación clínica.
