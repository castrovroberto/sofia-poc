from abc import ABC, abstractmethod

class LLMProvider(ABC):
    """Abstract base class for a Large Language Model provider."""

    @abstractmethod
    def get_embeddings(self):
        """Returns the embedding model for the provider."""
        pass

    @abstractmethod
    def get_llm(self):
        """Returns the language model for the provider."""
        pass
