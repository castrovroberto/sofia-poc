import os
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import config
from providers.base import LLMProvider
from providers.openai import OpenAIProvider
from providers.ollama import OllamaProvider
from loaders.pdf_loader import PDFLoader
from loaders.markdown_loader import MarkdownLoader
from loaders.url_loader import URLLoader

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

            if source_type in self.loaders:
                if os.path.exists(source_path):
                    logging.info(f"Loading documents from {source_path} using {source_type} loader...")
                    try:
                        loader = self.loaders[source_type]
                        documents = loader.load(source_path)
                        all_documents.extend(documents)
                    except Exception as e:
                        logging.error(f"Error loading {source_path}: {e}")
                else:
                    logging.warning(f"Source path not found: {source_path}")
            else:
                logging.warning(f"Unsupported source type: {source_type}")

        if not all_documents:
            logging.warning("No documents were loaded. Please check the sources in your config file.")
        else:
            logging.info(f"Loaded a total of {len(all_documents)} documents.")
            
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