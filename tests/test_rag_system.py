import pytest
from unittest.mock import MagicMock, patch, call
from rag_system import RAGSystem, get_provider
from providers.base import LLMProvider

@pytest.fixture
def mock_provider():
    """Fixture to create a mock LLMProvider."""
    provider = MagicMock(spec=LLMProvider)
    provider.get_embeddings.return_value = MagicMock()
    provider.get_llm.return_value = MagicMock()
    return provider

@patch('rag_system.os.path.exists')
def test_rag_system_setup_no_documents(mock_exists, mock_provider):
    """Test that RAGSystem.setup() returns False if no sources are found."""
    mock_exists.return_value = False
    rag_system = RAGSystem(mock_provider)
    assert not rag_system.setup()

def test_get_provider_openai(monkeypatch):
    """Test that get_provider() returns an OpenAIProvider when LLM_PROVIDER is 'openai'."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with patch('providers.openai.OpenAIProvider') as mock_openai_provider:
        get_provider()
        mock_openai_provider.assert_called_once()

def test_get_provider_ollama(monkeypatch):
    """Test that get_provider() returns an OllamaProvider when LLM_PROVIDER is 'ollama'."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    with patch('providers.ollama.OllamaProvider') as mock_ollama_provider:
        get_provider()
        mock_ollama_provider.assert_called_once()

def test_get_provider_unsupported(monkeypatch):
    """Test that get_provider() raises a ValueError for an unsupported provider."""
    monkeypatch.setenv("LLM_PROVIDER", "unsupported_provider")
    with pytest.raises(ValueError):
        get_provider()

@patch('rag_system.os.path.exists')
@patch('rag_system.PDFLoader')
@patch('rag_system.MarkdownLoader')
@patch('rag_system.URLLoader')
@patch('rag_system.FAISS')
@patch('rag_system.RetrievalQA')
def test_rag_system_happy_path(
    mock_retrieval_qa, mock_faiss, mock_url_loader, mock_md_loader, mock_pdf_loader, mock_exists, mock_provider
):
    """Test the full, successful execution of the RAGSystem setup and query process with multiple sources."""
    # --- Arrange ---
    # Mock file system and document loaders
    mock_exists.side_effect = lambda path: path in ['/app/pdfs', '/app/markdown', '/app/urls.txt']
    mock_pdf_loader.return_value.load.return_value = [MagicMock()]
    mock_md_loader.return_value.load.return_value = [MagicMock()]
    mock_url_loader.return_value.load.return_value = [MagicMock()]

    # Mock FAISS vector store
    mock_faiss.from_documents.return_value = MagicMock()

    # Mock the QA chain
    mock_qa_chain = MagicMock()
    mock_qa_chain.invoke.return_value = {'result': 'This is the answer.', 'source_documents': []}
    mock_retrieval_qa.from_chain_type.return_value = mock_qa_chain

    # --- Act ---
    rag_system = RAGSystem(mock_provider)
    setup_result = rag_system.setup()
    answer, sources = rag_system.ask("What is the answer?")

    # --- Assert ---
    assert setup_result is True
    assert answer == 'This is the answer.'
    assert sources == []

    # Verify that the key methods were called
    mock_pdf_loader.return_value.load.assert_called_once()
    mock_md_loader.return_value.load.assert_called_once()
    mock_url_loader.return_value.load.assert_called_once()
    mock_faiss.from_documents.assert_called_once()
    mock_retrieval_qa.from_chain_type.assert_called_once()
    mock_qa_chain.invoke.assert_called_once_with({'query': 'What is the answer?'})