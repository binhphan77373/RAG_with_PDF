# Document Query System

This repository contains a Python script for querying academic PDFs using natural language processing. The system is specifically configured to answer questions about the research paper.

## Overview

This application creates a question-answering system that:

1. Processes and chunks a PDF document
2. Embeds the chunks using HuggingFace embeddings
3. Stores the embeddings in a Chroma vector database
4. Answers user queries about the document using a language model

The system uses a retrieval-augmented generation approach to provide accurate answers based on the content of the academic paper.

## Features

- PDF document loading and preprocessing
- Text chunking for efficient processing
- Vector database storage using ChromaDB
- Document retrieval based on semantic similarity
- Natural language query processing
- Educational chatbot responses with context-aware answers
- Iterative answer refinement for improved accuracy

## Requirements

- Python 3.9+
- Required libraries listed in `requirements.txt` and `environment.yml`

## Installation

### Option 1: Using pip (requirements.txt)

1. Clone this repository
2. Create a new Python virtual environment:
```bash
# Create a virtual environment
python -m venv hitnet_env

# Activate the virtual environment
# On Windows
hitnet_env\Scripts\activate
# On macOS/Linux
source hitnet_env/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Option 2: Using Conda (environment.yml)

1. Clone this repository
2. Create a new conda environment:
```bash
conda env create -f environment.yml
```

3. Activate the environment:
```bash
conda activate hitnet-query
```

### Final Setup

Create a `.env` file in the root directory with your API keys:
```
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

## Usage

1. Place your PDF document in the root directory (default is "HITNet Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching.pdf")
2. Run the script:
```bash
python query_hitnet.py
```
3. The script will process the document and answer the default question: "What is HITNet?"

To customize queries, modify the `question` variable in the script.

## How It Works

1. **Document Preprocessing**: The PDF is loaded and split into manageable chunks.
2. **Embedding Generation**: Each chunk is converted into vector embeddings using HuggingFace's sentence transformer model.
3. **Vector Database**: The embeddings are stored in a Chroma vector database for efficient retrieval.
4. **Query Processing**: When a question is asked, the system finds the most relevant chunks in the database.
5. **Answer Generation**: The T5 language model generates an answer based on the retrieved context.
6. **Answer Refinement**: The system can refine answers by iteratively considering additional context.

## Configuration Options

The script includes several configurable parameters:
- `model_id`: Currently using "google/flan-t5-base"
- `chunk_size`: Text chunk size (currently 500 characters)
- `chunk_overlap`: Overlap between chunks (currently 100 characters)
- `embedding_model`: Currently using "sentence-transformers/all-MiniLM-L6-v2"
- `k`: Number of chunks to retrieve for each query (currently 2)

## Troubleshooting

Common issues:
- **GPU Memory Issues**: If you encounter GPU memory errors, modify the script to use CPU instead by setting `device='cpu'` in the appropriate places
- **Package Conflicts**: If you encounter dependency conflicts, try creating a clean environment using the provided environment.yml file
- **PDF Loading Errors**: Ensure your PDF is not password-protected and is readable

## Future Improvements

- Add a command-line interface for interactive querying
- Support for multiple document types beyond PDFs
- Improved prompt templates for better answer quality
- Web interface for easier interaction
- Batch processing of multiple documents
