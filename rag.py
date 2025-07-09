import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Configuración ---
DATA_PATH = "data"
DB_PATH = "vectordb"
MODEL_NAME = "llama3.2-vision:latest"  # Puedes cambiarlo por cualquier modelo que tengas en Ollama
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def create_vector_db():
    """Crea o carga la base de datos vectorial con un proceso más visual."""
    import glob
    
    print("--- INICIANDO PROCESO DE CREACIÓN DE BASE DE DATOS ---")
    
    # --- PASO 1: Encontrar archivos en la carpeta 'data' ---
    print("\nPaso 1: Buscando archivos .txt y .pdf en el directorio 'data'...")
    all_files = glob.glob("data/**/*", recursive=True)
    if not all_files:
        print("¡Error! No se encontraron archivos en la carpeta 'data'.")
        return None
    
    print(f"Archivos encontrados ({len(all_files)}):")
    for f in all_files:
        print(f"- {f}")

    # --- PASO 2: Cargar los documentos en memoria ---
    print("\nPaso 2: Cargando documentos en memoria...")
    loader = DirectoryLoader(DATA_PATH, glob="**/*", show_progress=True, use_multithreading=True)
    documents = loader.load()
    
    if not documents:
        print("¡Error! No se pudo cargar ningún documento. Revisa que los archivos no estén corruptos.")
        return None

    print(f"\n¡Carga completada! Se han procesado {len(documents)} páginas/documentos en total.")
    print("\n--- Contenido Extraído de los Documentos ---")
    for i, doc in enumerate(documents):
        print(f"\n--- Documento {i+1}: {doc.metadata.get('source', 'Desconocido')} (Página: {doc.metadata.get('page', 'N/A')}) ---")
        print(doc.page_content)
        print("----------------------------------------------------------------------------------------------------")
    print("\n--- Fin Contenido Extraído ---")
    
    # --- PASO 3: Dividir documentos en fragmentos (chunks) ---
    print("\nPaso 3: Dividiendo los documentos en fragmentos más pequeños (chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    print(f"Documentos divididos en {len(texts)} fragmentos de texto.")

    # --- PASO 4: Crear embeddings y guardar en la base de datos ---
    print(f"\nPaso 4: Creando representaciones numéricas (embeddings) con el modelo '{EMBEDDING_MODEL_NAME}'...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("Guardando los fragmentos y sus embeddings en la base de datos ChromaDB...")
    db = Chroma.from_documents(texts, embeddings, persist_directory=DB_PATH)
    db.persist()
    
    print("\n--- ¡PROCESO COMPLETADO! ---")
    print("Base de datos vectorial creada exitosamente en el directorio 'vectordb'.")
    return db

def main():
    """Función principal para ejecutar el RAG."""
    if not os.path.exists(DB_PATH):
        print("No se encontró una base de datos vectorial. Creando una nueva...")
        create_vector_db()
    
    print("Cargando base de datos vectorial existente...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 6}) # Aumentamos a 6 fragmentos

    print(f"Cargando LLM desde Ollama: {MODEL_NAME}...")
    llm = OllamaLLM(model=MODEL_NAME)

    # Definimos un prompt personalizado para guiar al LLM
    QA_CHAIN_PROMPT = PromptTemplate.from_template(
        """Eres un asistente experto en documentos. Utiliza únicamente la siguiente información proporcionada en el contexto para responder a la pregunta. Si la respuesta no se encuentra en el contexto, di 'No tengo información sobre eso en los documentos proporcionados.' No inventes respuestas.\n\nContexto: {context}\n\nPregunta: {question}\n\nRespuesta:"""
    )

    # Creando la cadena de RetrievalQA con el prompt personalizado
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    print("\n--- RAG Local Listo ---")
    print("Escribe tu pregunta o 'salir' para terminar.\n")

    while True:
        query = input("Pregunta: ")
        if query.lower() == 'salir':
            break
        if query.strip() == "":
            continue

        print("Procesando...")
        result = qa_chain({"query": query})
        
        print("\nRespuesta:")
        print(result["result"])
        print("\nDocumentos fuente recuperados:")
        if result["source_documents"]:
            for doc in result["source_documents"]:
                print(f"- {doc.metadata['source']} (página: {doc.metadata.get('page', 'N/A')})")
        else:
            print("- Ninguno.")
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()