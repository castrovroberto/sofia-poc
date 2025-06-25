#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print messages
print_message() {
    echo -e "${GREEN}[Sofia-PoC]${NC} $1"
}

print_error() {
    echo -e "${RED}[Error]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[Warning]${NC} $1"
}

# Function to validate OpenAI API key
validate_openai_key() {
    local key=$1
    # Check if key is empty
    if [ -z "$key" ]; then
        return 1
    fi
    # Check if key starts with "sk-" (OpenAI API key format)
    if [[ ! $key =~ ^sk- ]]; then
        return 1
    fi
    return 0
}

# Function to ensure directories exist
ensure_directories() {
    # Create necessary directories if they don't exist
    mkdir -p pdfs
    mkdir -p faiss_index_openai
    mkdir -p faiss_index_ollama
    mkdir -p providers
    mkdir -p loaders

    # Check if pdfs directory is empty
    if [ -z "$(ls -A pdfs 2>/dev/null)" ]; then
        print_warning "The 'pdfs' directory is empty. You should add some PDF or Markdown files to it."
        read -p "Do you want to continue anyway? (y/n): " continue_empty
        if [ "$continue_empty" != "y" ]; then
            exit 1
        fi
    fi
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Ensure all necessary directories exist
ensure_directories

# Build the Docker image
print_message "Building Docker image..."
if ! docker build -t rag-system .; then
    print_error "Failed to build Docker image."
    exit 1
fi
print_message "Docker image built successfully!"

# Prompt for provider selection
echo
print_message "Please select your LLM provider:"
echo "1) OpenAI (requires API key)"
echo "2) Ollama (requires local Ollama installation)"
read -p "Enter your choice (1 or 2): " provider_choice

case $provider_choice in
    1)
        # OpenAI setup
        if [ -z "$OPENAI_API_KEY" ]; then
            print_warning "OPENAI_API_KEY environment variable not found."
            while true; do
                read -p "Please enter your OpenAI API key: " api_key
                if validate_openai_key "$api_key"; then
                    export OPENAI_API_KEY=$api_key
                    break
                else
                    print_error "Invalid API key format. OpenAI API keys should start with 'sk-'"
                fi
            done
        else
            if ! validate_openai_key "$OPENAI_API_KEY"; then
                print_error "The OPENAI_API_KEY environment variable contains an invalid API key format."
                print_error "OpenAI API keys should start with 'sk-'"
                exit 1
            fi
        fi
        
        print_message "Starting container with OpenAI provider..."
        docker run -it --rm \
            -e OPENAI_API_KEY="$OPENAI_API_KEY" \
            -v "$(pwd)/pdfs":/app/pdfs \
            -v "$(pwd)/faiss_index_openai":/app/faiss_index_openai \
            rag-system
        ;;
        
    2)
        # Check if Ollama is running
        if ! curl -s http://localhost:11434/api/tags > /dev/null; then
            print_error "Ollama is not running. Please start Ollama and try again."
            print_message "You can start Ollama by running 'ollama serve' in a separate terminal."
            exit 1
        fi
        
        print_message "Starting container with Ollama provider..."
        docker run -it --rm \
            -e LLM_PROVIDER="ollama" \
            -v "$(pwd)/pdfs":/app/pdfs \
            -v "$(pwd)/faiss_index_ollama":/app/faiss_index_ollama \
            rag-system
        ;;
        
    *)
        print_error "Invalid choice. Please run the script again and select 1 or 2."
        exit 1
        ;;
esac
