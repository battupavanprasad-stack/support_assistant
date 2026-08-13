
# Zepto Support Assistant

## Overview

This project is a customer support assistant for Zepto.
It uses Retrieval-Augmented Generation (RAG) to answer questions
about Zepto policies using eight policy documents.

## Technologies Used

- Python
- Sentence Transformers
- all-MiniLM-L6-v2
- ChromaDB
- LangGraph
- FastAPI
- Pydantic
- Docker

## Architecture

The application follows this workflow:

User Question
    ↓
Intent Classification
    ↓
Policy Question or General Question
    ↓
Policy Question → ChromaDB Retrieval
General Question → Direct Answer
    ↓
Top 3 Relevant Documents
    ↓
MOCK_LLM Response
    ↓
Pydantic Response
    ↓
FastAPI /ask

## RAG

The eight Zepto policy documents are stored in the docs folder.

The all-MiniLM-L6-v2 model converts the documents into
384-dimensional embeddings.

The embeddings are stored in ChromaDB.

When a user asks a policy question, the question is converted
into an embedding and ChromaDB retrieves the top 3 most similar
documents using cosine similarity.

The retrieved information is then used to generate the answer.

## LangGraph

The LangGraph workflow contains three nodes:

1. classify_intent
2. retrieve_and_answer
3. direct_answer

Policy questions are routed to retrieve_and_answer.

General questions are routed to direct_answer.

## MOCK_LLM

The project uses MOCK_LLM by default.

When MOCK_LLM is not set or is set to 1, the application uses
a deterministic mock response and does not require an external
LLM API key.

This allows the required baseline workflow to run without
a paid API or external LLM service.

## API

### POST /ask

Example request:

{
    "query": "How long does Zepto delivery take?"
}

Example response:

{
    "answer": "Based on the retrieved context: ...",
    "sources": ["doc_01.txt"],
    "confidence": 1.0
}

### General Question Example

Request:

{
    "query": "What is your name?"
}

Response:

{
    "answer": "I can only answer questions about Zepto policies right now.",
    "sources": [],
    "confidence": 1.0
}

## Running the Application

Install the dependencies:

pip install -r requirements.txt

Start FastAPI:

uvicorn main:app --host 0.0.0.0 --port 7860

## Docker

Build the image:

docker build -t zepto-support-assistant .

Run the container:

docker run -p 7860:7860 zepto-support-assistant
