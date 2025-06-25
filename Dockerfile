# Use a slim Python base image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY rag_system.py .
COPY config.py .
COPY providers/ ./providers/
COPY loaders/ ./loaders/

# Create the directories for the sources and the FAISS index.
# We will mount host volumes to these paths at runtime.
RUN mkdir -p pdfs markdown faiss_index_openai faiss_index_ollama

# Copy the urls.txt file if it exists
COPY urls.txt .

# Command to run the application when the container starts
# Using -u option for unbuffered output, helpful for seeing logs in real-time
CMD ["python", "-u", "rag_system.py"]