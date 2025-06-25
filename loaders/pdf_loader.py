from .base import DocumentLoader
from langchain_community.document_loaders import PyPDFLoader

class PDFLoader(DocumentLoader):
    """Loads PDF documents."""

    def load(self, path: str):
        loader = PyPDFLoader(path)
        return loader.load()
