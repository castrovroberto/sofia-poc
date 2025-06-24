import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Provider-specific imports ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

if LLM_PROVIDER == "openai":
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
elif LLM_PROVIDER == "ollama":
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import Ollama
else:
    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. Please use 'openai' or 'ollama'.")

# --- Configuration ---
PDF_DIR = "/app/pdfs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Dynamic Configuration based on Provider ---
if LLM_PROVIDER == "openai":
    print("Using OpenAI provider.")
    EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
    VECTOR_DB_PATH = "/app/faiss_index_openai"
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable not set for OpenAI provider.")
else: # ollama
    print("Using Ollama provider.")
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    VECTOR_DB_PATH = "/app/faiss_index_ollama"

# --- 1. Document Loading ---
def load_documents(directory):
    print(f"Loading documents from {directory}...")
    # Supported file types and their loaders
    loaders = {
        ".pdf": lambda path: PyPDFLoader(path),
        ".md": lambda path: UnstructuredMarkdownLoader(path),
    }
    
    all_documents = []
    
    # Find all files in the directory
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            if file_ext in loaders:
                try:
                    print(f"Loading document: {file_path}")
                    loader = loaders[file_ext](file_path)
                    documents = loader.load()
                    all_documents.extend(documents)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    if not all_documents:
        print("No documents were loaded. Please check the directory and file types.")
    else:
        print(f"Loaded a total of {len(all_documents)} documents.")
        
    return all_documents

# --- 2. Text Splitting ---
def split_documents(documents, chunk_size, chunk_overlap):
    print(f"Splitting documents into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks

# --- 3. Text Embedding & 4. Vector Store Creation/Loading ---
def create_or_load_vector_store(chunks, vector_db_path, provider, embedding_model):
    print(f"Initializing embeddings for {provider}...")
    if provider == "openai":
        embeddings = OpenAIEmbeddings(model=embedding_model)
    else: # ollama
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    if os.path.exists(vector_db_path) and os.listdir(vector_db_path):
        print(f"Loading existing FAISS vector store from {vector_db_path}...")
        vector_store = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
        print("Vector store loaded.")
    else:
        print(f"Creating new FAISS vector store at {vector_db_path}...")
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(vector_db_path)
        print("Vector store created and saved.")
    return vector_store

# --- 5. & 6. Retrieval and LLM Setup (RAG Chain) ---
def setup_rag_chain(vector_store, provider):
    print(f"Setting up RAG chain with {provider}...")

    if provider == "openai":
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        print("Using OpenAI ChatOpenAI.")
    else: # ollama
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        llm = Ollama(base_url=ollama_base_url, model="llama2") # Default to llama2, can be changed
        print(f"Using Ollama from {ollama_base_url} with model 'llama2'.")

    qa_prompt = PromptTemplate(
        template="""Use the following context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:""",
        input_variables=["context", "question"],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt}
    )
    print("RAG chain setup complete.")
    return qa_chain

# --- Main Execution ---
def main():
    if not os.path.exists(PDF_DIR):
        print(f"Error: PDF directory '{PDF_DIR}' not found inside the container.")
        print("Please ensure you mount your local 'pdfs' directory to '/app/pdfs' when running the Docker container.")
        return

    documents = load_documents(PDF_DIR)
    if not documents:
        print(f"No PDF documents found in '{PDF_DIR}'. Please add some PDFs to the directory and ensure they are mounted.")
        print("Exiting.")
        return

    chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    vector_store = create_or_load_vector_store(chunks, VECTOR_DB_PATH, LLM_PROVIDER, EMBEDDING_MODEL_NAME)
    qa_chain = setup_rag_chain(vector_store, LLM_PROVIDER)

    print("\n--- RAG System Ready ---")
    print(f"--- Provider: {LLM_PROVIDER.upper()} ---")
    print("You can now ask questions about your PDFs.")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        query = input("\nYour question: ")
        if query.lower() in ["exit", "quit"]:
            break

        try:
            result = qa_chain.invoke({"query": query})
            print(f"\nAnswer: {result['result']}")
            print("\nSources:")
            for doc in result['source_documents']:
                print(f"- Page: {doc.metadata.get('page')}, Source: {doc.metadata.get('source')}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()