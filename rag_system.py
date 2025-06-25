import os
import warnings
import logging
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from config import SOURCES, CHUNK_SIZE, CHUNK_OVERLAP, PROMPT_TEMPLATE

import config
from providers.base import LLMProvider
from providers.openai import OpenAIProvider
from providers.ollama import OllamaProvider
from loaders.pdf_loader import PDFLoader
from loaders.markdown_loader import MarkdownLoader
from loaders.url_loader import URLLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Filter out specific deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pypdf')

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
    logger.info("Using OpenAI provider.")
    EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
    VECTOR_DB_PATH = "/app/faiss_index_openai"
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable not set for OpenAI provider.")
else: # ollama
    logger.info("Using Ollama provider.")
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    VECTOR_DB_PATH = "/app/faiss_index_ollama"

class RAGSystem:
    """A class to encapsulate the entire RAG system logic."""

    def __init__(self, provider: LLMProvider):
        """
        Initializes the RAG system with a specific provider.
        """
        self.provider = provider
        self.vector_store = None
        self.qa_chain = None
        self.embeddings = self.provider.get_embeddings()
        self.llm = self.provider.get_llm()
        self.loaders = {
            "pdf": PDFLoader(),
            "markdown": MarkdownLoader(),
            "url": URLLoader(),
        }

        # --- Configuration ---
        self.SOURCES = config.SOURCES
        self.CHUNK_SIZE = config.CHUNK_SIZE
        self.CHUNK_OVERLAP = config.CHUNK_OVERLAP
        self.PROMPT_TEMPLATE = config.PROMPT_TEMPLATE
        self.VECTOR_DB_PATH = f"/app/faiss_index_{self.provider.__class__.__name__.lower().replace('provider', '')}"

    def _load_documents(self):
        """Loads documents from the specified sources."""
        all_documents = []
        for source in self.SOURCES:
            source_type = source["type"]
            source_path = source["path"]
            is_recursive = source.get("recursive", True)

            if not os.path.exists(source_path):
                logger.warning(f"Source path not found: {source_path}")
                continue
            
            try:
                logger.info(f"Loading documents from {source_path} using {source_type} loader...")
                
                if source_type == "pdf":
                    if os.path.isdir(source_path):
                        loader = DirectoryLoader(
                            source_path,
                            glob="**/*.pdf" if is_recursive else "*.pdf",
                            loader_cls=PyPDFLoader
                        )
                        docs = loader.load()
                        all_documents.extend(docs)
                        logger.info(f"Loaded {len(docs)} PDF documents from {source_path}")
                
                elif source_type == "markdown":
                    if os.path.isdir(source_path):
                        loader = DirectoryLoader(
                            source_path,
                            glob="**/*.md" if is_recursive else "*.md",
                            loader_cls=UnstructuredMarkdownLoader
                        )
                        docs = loader.load()
                        all_documents.extend(docs)
                        logger.info(f"Loaded {len(docs)} Markdown documents from {source_path}")
                
            except Exception as e:
                logger.error(f"Error loading from {source_path}: {str(e)}")
        
        if not all_documents:
            logger.warning("No documents were loaded. Please check that your source directories contain supported files.")
        
        return all_documents

    def _split_documents(self, documents):
        """Splits documents into manageable chunks."""
        logging.info(f"Splitting documents into chunks (size={self.CHUNK_SIZE}, overlap={self.CHUNK_OVERLAP})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        logging.info(f"Created {len(chunks)} chunks.")
        return chunks

    def _create_or_load_vector_store(self, chunks):
        """Creates a new vector store or loads an existing one."""
        if os.path.exists(self.VECTOR_DB_PATH) and os.listdir(self.VECTOR_DB_PATH):
            logging.info(f"Loading existing FAISS vector store from {self.VECTOR_DB_PATH}...")
            self.vector_store = FAISS.load_local(self.VECTOR_DB_PATH, self.embeddings, allow_dangerous_deserialization=True)
            logging.info("Vector store loaded.")
        else:
            logging.info(f"Creating new FAISS vector store at {self.VECTOR_DB_PATH}...")
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            self.vector_store.save_local(self.VECTOR_DB_PATH)
            logging.info("Vector store created and saved.")

    def _setup_rag_chain(self):
        """Sets up the RAG retrieval chain."""
        logging.info(f"Setting up RAG chain with {self.provider.__class__.__name__}...")
        qa_prompt = PromptTemplate(
            template=self.PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": qa_prompt}
        )
        logging.info("RAG chain setup complete.")

    def setup(self):
        """Orchestrates the setup process."""
        documents = self._load_documents()
        if not documents:
            return False
            
        chunks = self._split_documents(documents)
        self._create_or_load_vector_store(chunks)
        self._setup_rag_chain()
        return True

    def ask(self, query: str):
        """
        Asks a question to the RAG system.
        """
        if not self.qa_chain:
            return "RAG chain is not set up. Please run setup() first.", []
        
        try:
            result = self.qa_chain.invoke({"query": query})
            return result['result'], result['source_documents']
        except Exception as e:
            logging.error(f"An error occurred during query processing: {e}")
            return f"An error occurred: {e}", []

    def start_cli(self):
        """Starts an interactive command-line interface for asking questions."""
        print("\n--- RAG System Ready ---")
        print(f"--- Provider: {self.provider.__class__.__name__} ---")
        print("You can now ask questions about your documents.")
        print("Type 'exit' or 'quit' to stop.")

        while True:
            query = input("\nYour question: ")
            if query.lower() in ["exit", "quit"]:
                break
            
            answer, sources = self.ask(query)
            print(f"\nAnswer: {answer}")
            if sources:
                print("\nSources:")
                for doc in sources:
                    page = doc.metadata.get('page', 'N/A')
                    source = doc.metadata.get('source', 'N/A')
                    print(f"- Page: {page}, Source: {source}")

def get_provider() -> LLMProvider:
    """Returns the appropriate LLM provider based on the environment variable."""
    provider_name = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider_name == "openai":
        return OpenAIProvider()
    elif provider_name == "ollama":
        return OllamaProvider()
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider_name}. Please use 'openai' or 'ollama'.")

def main():
    """Main execution function."""
    try:
        provider = get_provider()
        rag_system = RAGSystem(provider)
        if rag_system.setup():
            rag_system.start_cli()
    except ValueError as e:
        logging.error(e)

if __name__ == "__main__":
    main()