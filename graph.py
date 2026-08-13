
import os
from typing import TypedDict

import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END

from models import SupportResponse


CHROMA_PATH = "/content/support_assistant/chroma_db"


# Load the embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# Connect to ChromaDB
client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_or_create_collection(
    name="zepto_policies",
    metadata={"hnsw:space": "cosine"}
)


# State used by LangGraph
class SupportState(TypedDict):
    question: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float


# -----------------------------------------
# Intent classification
# -----------------------------------------

def classify_intent(question: str) -> str:

    policy_keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours"
    ]

    question = question.lower()

    for keyword in policy_keywords:
        if keyword in question:
            return "policy_question"

    return "general_question"


# -----------------------------------------
# Mock LLM
# -----------------------------------------

def mock_llm(question: str, context: list[str]) -> str:

    if not context:
        return (
            "I could not find enough information in the "
            "available Zepto policies."
        )

    return (
        f"Based on the retrieved context: {context[0]}"
    )


# -----------------------------------------
# Retrieve and answer
# -----------------------------------------

def retrieve_and_answer(question: str) -> SupportResponse:

    question_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    documents = results["documents"][0]
    document_ids = results["ids"][0]

    mock_mode = os.getenv(
        "MOCK_LLM", "1"
    ) == "1"

    if mock_mode:
        answer = mock_llm(
            question,
            documents
        )
    else:
        answer = mock_llm(
            question,
            documents
        )

    return SupportResponse(
        answer=answer,
        sources=document_ids,
        confidence=1.0
    )


# -----------------------------------------
# Direct answer
# -----------------------------------------

def direct_answer() -> SupportResponse:

    return SupportResponse(
        answer=(
            "I can only answer questions about "
            "Zepto policies right now."
        ),
        sources=[],
        confidence=1.0
    )


# -----------------------------------------
# LangGraph nodes
# -----------------------------------------

def classify_intent_node(state: SupportState):

    intent = classify_intent(
        state["question"]
    )

    return {
        "intent": intent
    }


def retrieve_and_answer_node(
    state: SupportState
):

    response = retrieve_and_answer(
        state["question"]
    )

    return {
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence
    }


def direct_answer_node(
    state: SupportState
):

    response = direct_answer()

    return {
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence
    }


# -----------------------------------------
# Conditional routing
# -----------------------------------------

def route_question(state: SupportState):

    if state["intent"] == "policy_question":
        return "retrieve"

    return "direct"


# -----------------------------------------
# Build LangGraph
# -----------------------------------------

graph_builder = StateGraph(
    SupportState
)

graph_builder.add_node(
    "classify_intent",
    classify_intent_node
)

graph_builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer_node
)

graph_builder.add_node(
    "direct_answer",
    direct_answer_node
)

graph_builder.add_edge(
    START,
    "classify_intent"
)

graph_builder.add_conditional_edges(
    "classify_intent",
    route_question,
    {
        "retrieve": "retrieve_and_answer",
        "direct": "direct_answer"
    }
)

graph_builder.add_edge(
    "retrieve_and_answer",
    END
)

graph_builder.add_edge(
    "direct_answer",
    END
)


support_graph = graph_builder.compile()
