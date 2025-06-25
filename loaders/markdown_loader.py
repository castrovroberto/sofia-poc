from .base import DocumentLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader

class MarkdownLoader(DocumentLoader):
    """Loads Markdown documents."""

    def load(self, path: str):
        loader = UnstructuredMarkdownLoader(path)
        return loader.load()
