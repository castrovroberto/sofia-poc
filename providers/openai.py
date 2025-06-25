import os
from .base import LLMProvider
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

class OpenAIProvider(LLMProvider):
    """Provider for OpenAI models."""

    def __init__(self):
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY environment variable not set for OpenAI provider.")
        self.embedding_model_name = "text-embedding-ada-002"

    def get_embeddings(self):
        return OpenAIEmbeddings(model=self.embedding_model_name)

    def get_llm(self):
        return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
