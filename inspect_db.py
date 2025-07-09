import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Configuración (debe coincidir con la de rag.py)
DB_PATH = "vectordb"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def inspect_chroma_db():
    if not os.path.exists(DB_PATH):
        print(f"Error: La base de datos no existe en {DB_PATH}")
        return

    print(f"Cargando base de datos ChromaDB desde {DB_PATH}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    collection = db.get() # Obtiene todos los documentos de la colección

    print(f"\n--- Información de la Colección ---")
    print(f"Número total de documentos: {len(collection['ids'])}")

    print(f"\n--- Primeros 5 Documentos (ID, Contenido y Metadatos) ---")
    for i in range(min(5, len(collection['ids']))):
        doc_id = collection['ids'][i]
        doc_content = collection['documents'][i]
        doc_metadata = collection['metadatas'][i]
        
        print(f"\nID: {doc_id}")
        print(f"Contenido: {doc_content[:200]}...") # Muestra los primeros 200 caracteres
        print(f"Metadatos: {doc_metadata}")
        print("-" * 30)

    print("\n--- Fin de la Inspección ---")

if __name__ == "__main__":
    inspect_chroma_db()
