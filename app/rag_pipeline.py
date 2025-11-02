import os
import glob
from dotenv import load_dotenv

# --- 1. CORRECCIÓN DE IMPORTACIONES DE LANGCHAIN ---
from langchain_core.callbacks import set_verbose, get_verbose
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.combine_documents import StuffDocumentsChain
import mlflow

load_dotenv()
set_verbose(True)

# --- CONFIGURACIÓN DE RUTAS ---
DATA_DIR = "data/pdfs"
PROMPT_DIR = "app/prompts"
VECTOR_DIR = "vectorstore"
VECTOR_INDEX_NAME = "faiss_index"

# --- CONFIGURACIÓN DE MODELOS ---
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-4o"

def load_documents(path=DATA_DIR):
    docs = []
    pdf_files = glob.glob(os.path.join(path, "*.pdf"))

    if not pdf_files:
        print(f"⚠️ No se encontraron archivos PDF en '{path}'.")
        return []

    for file_path in pdf_files:
        try:
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
        except Exception as e:
            print(f"❌ Error al cargar {file_path}: {e}")

    print(f"📄 {len(docs)} documentos cargados desde {len(pdf_files)} PDFs.")
    return docs


def save_vectorstore(chunk_size=512, chunk_overlap=50, persist_path=VECTOR_DIR):
    docs = load_documents()
    if not docs:
        print("No se generó el vectorstore porque no se encontraron documentos.")
        return

    # 1. Dividir texto
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)

    # 2. Crear embeddings y vectorstore
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)

    # 3. Guardar en disco
    full_path = os.path.join(persist_path, VECTOR_INDEX_NAME)
    os.makedirs(persist_path, exist_ok=True)
    vectordb.save_local(full_path)
    print(f"💾 Vectorstore guardado en: {full_path}")

    # 4. Registrar en MLflow
    mlflow.set_experiment("vectorstore_tracking")
    with mlflow.start_run(run_name="vectorstore_build"):
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("chunk_overlap", chunk_overlap)
        mlflow.log_param("n_chunks", len(chunks))
        mlflow.log_param("n_docs", len(docs))
        mlflow.set_tag("vectorstore_path", full_path)
        print(f"✅ MLflow Run ID: {mlflow.active_run().info.run_id}")


def load_vectorstore_from_disk(persist_path=VECTOR_DIR):
    full_path = os.path.join(persist_path, VECTOR_INDEX_NAME)
    if not os.path.exists(full_path + ".faiss"):
        raise FileNotFoundError(
            f"No se encontró el índice FAISS en '{full_path}.faiss'. Ejecuta 'save_vectorstore()' primero."
        )
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = FAISS.load_local(full_path, embeddings, allow_dangerous_deserialization=True)
    print("📦 Vectorstore cargado correctamente.")
    return vectordb


def load_prompt(version="v1_asistente_PM"):
    prompt_path = os.path.join(PROMPT_DIR, f"{version}.txt")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"No se encontró el prompt: {prompt_path}")

    with open(prompt_path, "r", encoding='utf-8') as f:
        system_prompt = f.read()

    template = (
        f"{system_prompt}\n\n"
        "Usando el siguiente contexto y el historial de conversación, responde la pregunta al final. "
        "Si no tienes suficiente información en el contexto, indica amablemente que no puedes responder basándote solo en los documentos.\n\n"
        "Contexto:\n{context}\n\n"
        "Historial:\n{chat_history}\n\n"
        "Pregunta: {question}\n"
        "Respuesta:"
    )

    return PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )


def build_chain(vectordb, prompt_version="v1_asistente_PM"):
    prompt = load_prompt(prompt_version)
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    # Crear la cadena de combinación
    combine_chain = StuffDocumentsChain(
        llm_chain=LLMChain(llm=llm, prompt=prompt),
        document_variable_name="context"
    )

    # Crear la cadena conversacional
    chain = ConversationalRetrievalChain(
        retriever=retriever,
        combine_docs_chain=combine_chain,
        return_source_documents=True
    )

    print("🤖 Cadena conversacional creada correctamente.")
    return chain


if __name__ == "__main__":
    print("🚀 Ejecutando la función de ingesta y guardado del vectorstore...")
    save_vectorstore()
