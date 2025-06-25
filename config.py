# --- RAG System Configuration ---

# -- File Paths --
# The directory where the source documents are located.
# This path is relative to the Docker container's file system.
SOURCES = [
    {"type": "pdf", "path": "/app/pdfs"},
    {"type": "markdown", "path": "/app/markdown"},
    {"type": "url", "path": "/app/urls.txt"}
]

# -- Text Splitting --
# The maximum size of a text chunk.
CHUNK_SIZE = 1000
# The number of characters to overlap between chunks.
CHUNK_OVERLAP = 200

# -- Prompt Template --
# The template for the prompt that is sent to the LLM.
PROMPT_TEMPLATE = """Use the following context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
