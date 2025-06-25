import os
from .base import LLMProvider
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

class OllamaProvider(LLMProvider):
    """Provider for Ollama models."""

    def __init__(self):
        self.embedding_model_name = "all-MiniLM-L6-v2"
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

    def get_embeddings(self):
        return HuggingFaceEmbeddings(model_name=self.embedding_model_name)

    def get_llm(self):
        return Ollama(base_url=self.ollama_base_url, model="llama2")
