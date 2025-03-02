import os
from dotenv import load_dotenv  # Load environment variables from a .env file

import textwrap  # For formatting text if needed later

# Import various libraries used in the script
import langchain
import chromadb
import transformers
import openai
import torch
import requests
import json

# Import specific classes and functions from libraries
from transformers import AutoTokenizer, T5ForConditionalGeneration, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# =============================================================================
# 1. Environment Setup and API Key Loading
# =============================================================================

# Load environment variables from the .env file (e.g., API keys)
load_dotenv()

# Retrieve tokens from environment variables
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
#openai_api_key = os.getenv("OPENAI_API_KEY")

# =============================================================================
# 2. Initialize the Language Model with HuggingFacePipeline (replacing Ollama)
# =============================================================================

# Initialize a HuggingFace model pipeline instead of OllamaLLM
model_id = "google/flan-t5-base"  # Using smaller model to avoid sequence length issues
tokenizer = AutoTokenizer.from_pretrained(model_id)
model_hf = T5ForConditionalGeneration.from_pretrained(model_id)

# Create a pipeline for text generation
pipe = pipeline(
    "text2text-generation",
    model=model_hf, 
    tokenizer=tokenizer, 
    max_length=512,  # Reduced max length
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)

# Create LangChain wrapper around the HuggingFace pipeline
model = HuggingFacePipeline(pipeline=pipe)

# =============================================================================
# 3. Document Preprocessing Function
# =============================================================================

def docs_preprocessing_helper(file):
    """
    Helper function to load and preprocess a PDF file containing data.

    This function performs two main tasks:
      1. Loads the PDF file using PyPDFLoader from LangChain.
      2. Splits the loaded documents into smaller text chunks using CharacterTextSplitter.
    
    Args:
        file (str): Path to the PDF file.
        
    Returns:
        list: A list of document chunks ready for embedding and indexing.
    """
    # Load the PDF file using LangChain's PyPDFLoader.
    loader = PyPDFLoader(file)
    docs = loader.load_and_split()
    
    # Create a text splitter that divides the documents into chunks up to 1000 characters
    # with an overlap of 200 characters between chunks.
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)  # Reduced chunk size
    docs = text_splitter.split_documents(docs)
    
    return docs

# Preprocess the PDF file and store the document chunks in 'docs'.
docs = docs_preprocessing_helper('HITNet Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching.pdf')

# =============================================================================
# 4. Set Up the Embedding Function and Chroma Database
# =============================================================================

# Initialize the embedding function
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}  # Added normalization
)

# Create a vector store (ChromaDB) from the document chunks using the embedding function.
db = Chroma.from_documents(
    documents=docs,
    embedding=embedding_function,
    persist_directory="my_chroma_db"
)

# =============================================================================
# 5. Define and Initialize the Prompt Template
# =============================================================================

# Define a prompt template that instructs the chatbot on how to answer queries.
template = """You are an educational chatbot specialized in explaining academic concepts.
Please use the provided context to answer the question.
If you cannot find the information or if it's incomplete, please say:
"I apologize, but I cannot find this information in the provided data."

Context:
{context}

Question: {question}
"""

# Create a PromptTemplate object from LangChain with the defined template.
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Format the prompt with a general context message.
formatted_prompt = prompt.format(
    context="You are interacting with college students. They will ask you questions related to the file provided. Please answer their specific questions using the provided file.",
    question="Explain the HitNet Model"
)

# Define a refine prompt for iterative refinement if new context is provided.
refine_prompt_template = """You are a teaching chatbot. We have an existing answer: 
{existing_answer}

We have the following new context to consider:
{context}

Please refine the original answer if there's new or better information. 
If the new context does not change or add anything to the original answer, keep it the same.

If the answer is not in the source data or is incomplete, say:
"I apologize, but I cannot find this information in the provided data."

Question: {question}

Refined Answer:
"""

refine_prompt = PromptTemplate(
    template=refine_prompt_template,
    input_variables=["existing_answer", "context", "question"]
)

# =============================================================================
# 6. Create the RetrievalQA Chain
# =============================================================================

# The RetrievalQA chain combines:
#   - The language model (model) to generate responses.
#   - A retriever (db.as_retriever) that fetches relevant document chunks based on the query.
#   - A prompt that provides instructions on how to answer the query.
chain_type_kwargs = {
    "question_prompt": prompt,
    "refine_prompt": refine_prompt,
    "document_variable_name": "context",
}

chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="refine",
    retriever=db.as_retriever(search_kwargs={"k": 2}),
    chain_type_kwargs=chain_type_kwargs,
)

# =============================================================================
# 7. Query the Chain and Output the Response
# =============================================================================

# Run the chain with the query. The chain will:
#   1. Retrieve the most relevant document chunk(s) from ChromaDB.
#   2. Format the prompt with that context.
#   3. Use the language model to generate an answer based on the prompt and retrieved data.
question = "What is HITNet?"  # Example question in English
response = chain.invoke({"query": question})

# Print the response to the console.
print(response)