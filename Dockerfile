# Use a slim Python base image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create all necessary directories
# These will be mount points for host volumes
RUN mkdir -p \
    pdfs \
    faiss_index_openai \
    faiss_index_ollama \
    providers \
    loaders

# Copy application files if they exist
COPY rag_system.py ./
COPY config.py ./
COPY providers/ ./providers/
COPY loaders/ ./loaders/

# Copy the urls.txt file if it exists
COPY urls.txt ./

# Add a healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3     CMD python -c "import rag_system"

# Command to run the application when the container starts
# Using -u option for unbuffered output, helpful for seeing logs in real-time
CMD ["python", "-u", "rag_system.py"]