# NCERT Class 10 Science Chatbot with Smart Caching

This project is a doubt-solving chatbot for the NCERT Class 10 Science textbook, built with a Retrieval-Augmented Generation (RAG) pipeline and a custom smart caching layer. It was developed as a take-home assignment for the AI Intern role at GlobusLearn Services (Prepzy.ai).

## Features
* **Doubt-Solving RAG:** Answers student questions based strictly on the NCERT Class 10 Science textbook and cites the relevant chapter.
* **Smart Semantic Caching:** Uses local embeddings (`all-MiniLM-L6-v2`) to check for similar past questions. If a semantic match is found (>0.95 similarity) and numerical values match exactly, it serves the cached answer instantly (sub-500ms latency) without calling the LLM.
* **Context-Aware Multi-turn:** Identifies follow-up questions (e.g., "What about its laws?") and bypasses the cache to maintain conversational flow using the LLM.
* **Free & Fast LLM:** Utilizes the Groq API for high-speed, cost-free inference.

## Tech Stack
* **Language:** Python
* **LLM Framework:** LangChain
* **Vector Store & Caching:** FAISS, Sentence-Transformers (HuggingFace)
* **Backend API:** FastAPI
* **Frontend UI:** Streamlit
* **LLM:** LLaMA-3 (via Groq API)

## Prerequisites
* Python 3.9+
* A free [Groq API Key](https://console.groq.com/)

## Setup & Installation

**1. Clone the repository**
```bash
git clone <YOUR_GITHUB_REPO_URL_HERE>
cd prepzy_Project
