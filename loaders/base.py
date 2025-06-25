from abc import ABC, abstractmethod

class DocumentLoader(ABC):
    """Abstract base class for a document loader."""

    @abstractmethod
    def load(self, path: str):
        """Loads documents from a given path."""
        pass
