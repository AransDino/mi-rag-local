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
    """Crea la base de datos vectorial tratando los documentos de forma más inteligente."""
    from langchain.docstore.document import Document
    import re
    import glob

    print("--- INICIANDO PROCESO DE CREACIÓN DE BASE DE DATOS ---")
    
    # --- PASO 1: Cargar documentos de forma personalizada ---
    print("\nPaso 1: Cargando y procesando archivos desde 'data'...")
    
    all_documents = []
    
    # Primero, procesamos el archivo especial de forma manual si existe
    special_file_path = os.path.join(DATA_PATH, "dataset_prueba_rag.txt")
    if os.path.exists(special_file_path):
        print(f"Procesando archivo especial: {special_file_path}")
        try:
            with open(special_file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Dividir por el separador de documentos
            parts = re.split(r'===\s*DOCUMENTO\s*\d+\s*===', content)
            
            doc_count = 0
            for part in parts:
                if part.strip():
                    doc_count += 1
                    all_documents.append(Document(page_content=part.strip(), metadata={"source": special_file_path, "document_part": doc_count}))
            
            print(f"'{special_file_path}' fue dividido en {doc_count} documentos semánticos.")
        except Exception as e:
            print(f"Error procesando el archivo especial: {e}")

    # Ahora, cargamos el resto de archivos (PDFs, otros .txt, etc.)
    print("\nCargando otros documentos (PDFs, etc.)...")
    other_files_pattern = os.path.join(DATA_PATH, "**/*")
    # Usamos DirectoryLoader pero excluimos el archivo que ya procesamos
    loader = DirectoryLoader(
        DATA_PATH, 
        glob="**/*", 
        show_progress=True, 
        use_multithreading=True,
        exclude=[special_file_path]
    )
    other_documents = loader.load()
    all_documents.extend(other_documents)

    if not all_documents:
        print("¡Error! No se pudo cargar ningún documento.")
        return None

    print(f"\n¡Carga completada! Se han procesado {len(all_documents)} documentos en total.")
    
    # --- PASO 2: Dividir documentos en fragmentos (chunks) ---
    print("\nPaso 2: Dividiendo los documentos en fragmentos más pequeños (chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    texts = text_splitter.split_documents(all_documents)
    print(f"Documentos divididos en {len(texts)} fragmentos de texto.")

    # --- PASO 3: Crear embeddings y guardar en la base de datos ---
    print(f"\nPaso 3: Creando representaciones numéricas (embeddings) con el modelo '{EMBEDDING_MODEL_NAME}'...")
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