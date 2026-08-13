
from fastapi import FastAPI

from models import SupportResponse, AskRequest
from graph import support_graph


app = FastAPI(
    title="Zepto Support Assistant",
    description="RAG based customer support assistant for Zepto"
)


@app.get("/")
def home():
    return {
        "message": "Zepto Support Assistant is running"
    }


@app.post(
    "/ask",
    response_model=SupportResponse
)
def ask_question(request: AskRequest):

    result = support_graph.invoke({
        "question": request.query
    })

    return SupportResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )
