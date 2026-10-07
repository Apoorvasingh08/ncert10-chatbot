import os
import re
import time
import uuid
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local dev; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# LLM + retriever setup
# ---------------------------------------------------------------------------


def get_llm():
    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq

        return ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
        )

    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

    return None


def get_embeddings():
    # Must match the model used when the FAISS index was built (ingest.py).
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


index_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index")
retriever = None

if os.path.isdir(index_dir):
    try:
        embeddings = get_embeddings()
        vectorstore = FAISS.load_local(
            index_dir, embeddings, allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        print("FAISS index loaded OK")
    except Exception as e:
        print("FAISS LOAD FAILED:", repr(e))
        retriever = None
else:
    print("faiss_index folder NOT FOUND at:", index_dir)

llm = get_llm()
print("LLM configured:", llm is not None)

system_prompt = (
    "You are a doubt-solving chatbot for NCERT Class 10 Science. "
    "Use only the provided context to answer. If the context does not contain the answer, "
    "politely decline. Cite the chapter name used at the end of your response.\n\n"
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# ---------------------------------------------------------------------------
# Semantic cache + sessions
# ---------------------------------------------------------------------------

cache_model = SentenceTransformer("all-MiniLM-L6-v2")
cache_store: List[Dict[str, Any]] = []
sessions: Dict[str, List[dict]] = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    citations: List[str]
    cache_hit: bool
    latency_ms: int


@app.post("/session")
def create_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = []
    return {"session_id": session_id}


def extract_numbers(text: str) -> set:
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))


def check_cache(query: str, history: List[dict]):
    # Only cache first questions of a conversation (no follow-up context).
    if history:
        return None

    query_nums = extract_numbers(query)
    query_emb = cache_model.encode(query, convert_to_tensor=True)

    best_match = None
    highest_score = 0.0

    for item in cache_store:
        score = util.cos_sim(query_emb, item["embedding"]).item()
        if score > 0.95:
            cached_nums = extract_numbers(item["query"])
            if query_nums == cached_nums and score > highest_score:
                highest_score = score
                best_match = item

    return best_match


# ---------------------------------------------------------------------------
# RAG answer
# ---------------------------------------------------------------------------


def answer_question(question: str):
    if retriever is None:
        raise HTTPException(
            status_code=500,
            detail="Vector index not available. Check the backend terminal for 'FAISS LOAD FAILED' "
            "or run 'python ingest.py' to build the faiss_index folder.",
        )

    if llm is None:
        raise HTTPException(
            status_code=500,
            detail="No LLM API key configured. Set GROQ_API_KEY in your .env file.",
        )

    try:
        docs = retriever.invoke(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    context = "\n\n".join(doc.page_content for doc in docs)
    formatted_prompt = prompt.invoke({"input": question, "context": context})

    try:
        response = llm.invoke(formatted_prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    answer = response.content if hasattr(response, "content") else str(response)

    citations = list(
        dict.fromkeys(
            os.path.basename(doc.metadata.get("source", "NCERT Textbook")).replace(".pdf", "")
            for doc in docs
        )
    )
    return answer, citations


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    start_time = time.time()
    session_id = request.session_id
    user_message = request.message

    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]

    cached_result = check_cache(user_message, history)

    if cached_result:
        latency = int((time.time() - start_time) * 1000)
        sessions[session_id].append({"user": user_message, "ai": cached_result["reply"]})
        return ChatResponse(
            reply=cached_result["reply"],
            citations=cached_result["citations"],
            cache_hit=True,
            latency_ms=latency,
        )

    # Keep only the last 4 turns so the prompt stays small and retrieval stays focused.
    recent = history[-4:]
    full_input = (
        "\n".join(f"User: {h['user']}\nAI: {h['ai']}" for h in recent)
        + f"\nUser: {user_message}"
    ).strip()

    reply, citations = answer_question(full_input)

    if not history:
        query_emb = cache_model.encode(user_message, convert_to_tensor=True)
        cache_store.append(
            {
                "query": user_message,
                "embedding": query_emb,
                "reply": reply,
                "citations": citations,
            }
        )

    sessions[session_id].append({"user": user_message, "ai": reply})

    latency = int((time.time() - start_time) * 1000)
    return ChatResponse(
        reply=reply,
        citations=citations,
        cache_hit=False,
        latency_ms=latency,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)