from .base import DocumentLoader
from langchain_community.document_loaders import WebBaseLoader

class URLLoader(DocumentLoader):
    """Loads documents from a list of URLs."""

    def load(self, path: str):
        with open(path, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        loader = WebBaseLoader(urls)
        return loader.load()
