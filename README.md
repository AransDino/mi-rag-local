# RAG Local con Ollama

Este proyecto implementa un sistema de Generación Aumentada por Recuperación (RAG) que se ejecuta completamente en tu máquina local. Utiliza Ollama para acceder a modelos de lenguaje grandes (LLMs) y una base de datos vectorial ChromaDB para permitir al LLM responder preguntas basándose en un conjunto de documentos proporcionados por el usuario.

## Características

- **100% Local**: No depende de APIs externas de pago.
- **Ollama Integrado**: Fácil de conectar con cualquier modelo que tengas en Ollama.
- **Soporte Multiformato**: Procesa archivos `.txt`, `.pdf` y otros gracias a la librería `unstructured`.
- **Persistente**: Crea una base de datos vectorial local para no tener que procesar los mismos archivos cada vez.

## Instalación

Sigue estos pasos para poner en marcha el proyecto.

1.  **Clonar el Repositorio** (si lo has descargado de otro sitio)
    ```bash
    git clone https://github.com/AransDino/mi-rag-local.git
    cd mi-rag-local
    ```

2.  **Crear un Entorno Virtual**
    Desde la carpeta del proyecto, ejecuta:
    ```bash
    python -m venv venv
    ```

3.  **Activar el Entorno Virtual**
    - En **Windows (PowerShell)**:
      ```powershell
      .\venv\Scripts\Activate.ps1
      ```
      *(Si encuentras un error, puede que necesites ejecutar `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` primero)*

    - En **Linux/macOS**:
      ```bash
      source venv/bin/activate
      ```

4.  **Instalar Dependencias**
    Asegúrate de que tu entorno virtual está activado (verás `(venv)` en la terminal) y luego ejecuta:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

1.  **Añadir Documentos**: Copia todos tus documentos (`.pdf`, `.txt`, etc.) en la carpeta `data`.

2.  **Ejecutar el Script**: La primera vez que lo ejecutes, creará la base de datos vectorial (`vectordb`). El script mostrará el proceso de carga y, cuando esté listo, podrás empezar a hacer preguntas.
    ```bash
    python rag.py
    ```

3.  **Actualizar la Base de Conocimiento**: Si en el futuro añades, modificas o eliminas archivos en la carpeta `data`, debes forzar la reconstrucción de la base de datos. Para ello:
    - **Borra la carpeta `vectordb`**.
    - Vuelve a ejecutar `python rag.py`.

4.  **Interactuar**: Escribe tus preguntas en la terminal y pulsa Enter. Para salir, escribe `salir`.

## Configuración

Puedes modificar las siguientes variables al principio del script `rag.py` para ajustar el comportamiento:

- `MODEL_NAME`: El nombre del modelo de Ollama que quieres usar (ej. `"llama3.2-vision:latest"`).
- `EMBEDDING_MODEL_NAME`: El modelo de Hugging Face para crear los embeddings.

## Historial de Cambios

### 09 de Julio de 2025 - Mejoras en la Extracción y Configuración del RAG

Esta actualización incluye varias mejoras y correcciones para optimizar el rendimiento del sistema RAG:

- **Actualización de Dependencias**: Se han actualizado las importaciones de LangChain a sus paquetes más recientes (`langchain-huggingface`, `langchain-ollama`) y se ha corregido `requirements.txt` para incluir todas las dependencias necesarias (`unstructured[pdf]`, `pypdf`).
- **Mejoras en la Extracción de Documentos**:
    - Se ha corregido un `SyntaxError` persistente en `rag.py` relacionado con la impresión del contenido extraído.
    - Se ha añadido una salida detallada (`--- Contenido Extraído de los Documentos ---`) durante la creación de la base de datos para visualizar el texto que `unstructured` extrae de los documentos (incluidos PDFs).
- **Optimización de la Recuperación (Retriever)**:
    - Se ha aumentado el número de fragmentos (`k`) que el retriever busca de `2` a `6` para proporcionar más contexto al LLM.
    - Se ha implementado un `Prompt Template` personalizado para guiar al LLM a responder únicamente con la información proporcionada en el contexto y a indicar explícitamente cuando la información no se encuentra.
- **Corrección de Inicialización del LLM**: Se ha corregido el `NameError` al inicializar el modelo de Ollama (`Ollama` a `OllamaLLM`).

## Herramientas Adicionales

### `inspect_db.py`

Este script permite inspeccionar el contenido de la base de datos vectorial `vectordb` que crea `rag.py`. Es útil para verificar qué documentos y fragmentos se han almacenado, así como sus metadatos.

**Uso:**

1.  Asegúrate de que `rag.py` haya ejecutado al menos una vez y haya creado la carpeta `vectordb`.
2.  Con tu entorno virtual activado, ejecuta:
    ```bash
    python inspect_db.py
    ```
3.  El script mostrará el número total de documentos en la base de datos y los primeros 5 documentos con su contenido parcial y metadatos.
