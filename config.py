# --- RAG System Configuration ---

# -- File Paths --
# The directory where the source documents are located.
# This path is relative to the Docker container's file system.
# Each source is a dictionary with:
#   - type: The type of documents to load ("pdf" or "markdown")
#   - path: The path to the documents
#   - recursive: Whether to search for documents recursively (default: True)
SOURCES = [
    {
        "type": "pdf",
        "path": "/app/pdfs",
        "recursive": True
    }
]

# -- Text Splitting --
# The maximum size of a text chunk.
CHUNK_SIZE = 1000
# The number of characters to overlap between chunks.
CHUNK_OVERLAP = 200

# -- Prompt Template --
# The template for the prompt that is sent to the LLM.
# Available variables:
#   - {context}: The relevant document chunks
#   - {question}: The user's question
PROMPT_TEMPLATE = """Use the following context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}
Helpful Answer:"""
