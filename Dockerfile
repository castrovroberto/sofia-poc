# Use a slim Python base image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rag_system.py script
COPY rag_system.py .

# Create the directory for PDFs and the FAISS index.
# We will mount host volumes to these paths at runtime.
RUN mkdir -p pdfs faiss_index

# Command to run the application when the container starts
# Using -u option for unbuffered output, helpful for seeing logs in real-time
CMD ["python", "-u", "rag_system.py"]