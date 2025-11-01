import os
import glob
from dotenv import load_dotenv

# --- 1. CORRECCIÓN DE IMPORTACIONES DE LANGCHAIN ---
# La importación de 'globals' fue movida a 'callbacks' en versiones recientes.
from langchain.callbacks import set_verbose, get_verbose

set_verbose(True) # Si quieres ver logs detallados

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

import mlflow

load_dotenv()

# --- CONFIGURACIÓN DE RUTAS ---
DATA_DIR = "data/pdfs"
PROMPT_DIR = "app/prompts"
VECTOR_DIR = "vectorstore"
VECTOR_INDEX_NAME = "faiss_index" # Nombre específico para el archivo

# --- CONFIGURACIÓN DE MODELOS ---
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-4o" 

def load_documents(path=DATA_DIR):
    """Carga todos los archivos PDF desde el directorio especificado."""
    docs = []
    # Usar glob para manejar rutas correctamente en diferentes SO
    pdf_files = glob.glob(os.path.join(path, "*.pdf"))
    
    if not pdf_files:
        print(f"ADVERTENCIA: No se encontraron archivos PDF en '{path}'.")
        return []

    for file_path in pdf_files:
        try:
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error al cargar el PDF {file_path}: {e}")
            
    return docs

def save_vectorstore(chunk_size: int = 512, chunk_overlap: int = 50, persist_path: str = VECTOR_DIR):
    """Procesa PDFs, crea el índice FAISS, lo guarda y registra en MLflow."""
    
    docs = load_documents()
    if not docs:
        print("No se generó el vectorstore porque no se encontraron documentos.")
        return

    # 1. Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    
    # 2. Embeddings y Vector Store
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)
    
    # 3. Guardar en disco
    full_path = os.path.join(persist_path, VECTOR_INDEX_NAME)
    if not os.path.exists(persist_path):
        os.makedirs(persist_path)
        
    # FAISS guarda dos archivos: .faiss y .pkl
    vectordb.save_local(full_path)
    print(f"Vector store guardado en: {full_path}")

    # 4. Registro MLflow
    mlflow.set_experiment("vectorstore_tracking")
    with mlflow.start_run(run_name="vectorstore_build"):
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("chunk_overlap", chunk_overlap)
        mlflow.log_param("n_chunks", len(chunks))
        mlflow.log_param("n_docs", len(docs))
        mlflow.set_tag("vectorstore_path", full_path)
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")


def load_vectorstore_from_disk(persist_path=VECTOR_DIR):
    """Carga el índice FAISS persistido desde el disco."""
    full_path = os.path.join(persist_path, VECTOR_INDEX_NAME)
    if not os.path.exists(full_path + ".faiss"):
        raise FileNotFoundError(
            f"El índice FAISS no se encontró en '{full_path}.faiss'. Ejecuta 'save_vectorstore()' primero."
        )
        
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    # allow_dangerous_deserialization=True es necesario para cargar el índice
    return FAISS.load_local(full_path, embeddings, allow_dangerous_deserialization=True)


def load_prompt(version="v1_asistente_PM"):
    """Carga y devuelve el PromptTemplate completo para la cadena RAG."""
    
    # Se recomienda crear un template único que incluya el historial de chat 
    # y el contexto para la ConversationalRetrievalChain.
    
    prompt_path = os.path.join(PROMPT_DIR, f"{version}.txt")
    if not os.path.exists(prompt_path):
        # Ajustado para usar el nombre del prompt de Política Monetaria
        raise FileNotFoundError(f"Prompt no encontrado: {prompt_path}. Asegúrate de que existe '{version}.txt'")
        
    with open(prompt_path, "r", encoding='utf-8') as f:
        system_prompt = f.read()

    # Template adaptado para ConversationalRetrievalChain que espera chat_history y context
    FULL_PROMPT_TEMPLATE = (
        f"{system_prompt}\n\n"
        "Usando el siguiente contexto y el historial de conversación, responde la pregunta al final. "
        "Si no tienes suficiente información en el contexto, indica amablemente que no puedes responder basándote solo en los documentos.\n\n"
        "Contexto:\n{context}\n\n"
        "Historial de conversación:\n{chat_history}\n\n"
        "Pregunta: {question}\n"
        "Respuesta:"
    )
    
    # ConversationalRetrievalChain usa estas tres variables internamente para el combine_docs_chain
    return PromptTemplate(
        input_variables=["context", "chat_history", "question"], 
        template=FULL_PROMPT_TEMPLATE
    )


def build_chain(vectordb, prompt_version="v1_asistente_PM"):
    """Construye y devuelve la ConversationalRetrievalChain."""
    
    prompt = load_prompt(prompt_version)
    retriever = vectordb.as_retriever()
    
    # Inicializar LLM
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    # Construir la cadena
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        # Pasar el prompt al combine_docs_chain_kwargs
        combine_docs_chain_kwargs={"prompt": prompt}, 
        return_source_documents=True # Es muy útil devolver las fuentes para el chatbot
    )

# Bloque para la ejecución directa
if __name__ == "__main__":
    print("Ejecutando la función de ingesta y guardado del vectorstore...")
    save_vectorstore()
