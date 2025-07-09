# Directrices para el Agente Gemini

## Objetivo del Proyecto

Implementar un sistema RAG (Retrieval-Augmented Generation) local con Ollama para responder preguntas basadas en documentos locales (PDF, TXT).

## Convenciones de Código

- **Lenguaje principal**: Python
- **Estilo de código**: PEP 8 (estándar para Python)
- **Gestor de dependencias**: pip con `requirements.txt`

## Directrices de Desarrollo

- Siempre usar entornos virtuales para aislar dependencias.
- Recordar que `unstructured[pdf]` es una dependencia crucial para el procesamiento de PDFs.
- Para actualizar la base de conocimiento, se debe borrar la carpeta `vectordb` y volver a ejecutar el script.
- Los modelos de Ollama pueden necesitar ser actualizados con `ollama pull <nombre_modelo>` si hay incompatibilidades.
- El script `rag.py` utiliza `DirectoryLoader(glob="**/*")` para cargar documentos, lo que requiere que las dependencias de los tipos de archivo (ej. `pypdf`, `unstructured[pdf]`) estén instaladas.

## Comandos Importantes

- **Instalar dependencias**: `pip install -r requirements.txt`
- **Ejecutar la aplicación**: `python rag.py`
- **Actualizar modelo Ollama**: `ollama pull <nombre_modelo>`
- **Borrar base de datos vectorial**: `Remove-Item -Path vectordb -Recurse -Force` (para PowerShell)
