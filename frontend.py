import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Prepzy Doubt Solver", page_icon="🤖")

if "session_id" not in st.session_state:
    response = requests.post(f"{API_URL}/session")
    st.session_state.session_id = response.json()["session_id"]
    st.session_state.messages = []

st.title("NCERT Class 10 Science Chatbot")

if st.button("New conversation"):
    response = requests.post(f"{API_URL}/session")
    st.session_state.session_id = response.json()["session_id"]
    st.session_state.messages = []
    st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            meta = f"**Citations:** {', '.join(msg['citations'])} | **Cache Hit:** {msg['cache_hit']} | **Latency:** {msg['latency']}ms"
            st.caption(meta)

if prompt := st.chat_input("Ask your doubt here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            res = requests.post(
                f"{API_URL}/chat",
                json={"session_id": st.session_state.session_id, "message": prompt}
            )
            
            if res.status_code == 200:
                data = res.json()
                st.markdown(data["reply"])
                
                meta = f"**Citations:** {', '.join(data['citations'])} | **Cache Hit:** {data['cache_hit']} | **Latency:** {data['latency_ms']}ms"
                st.caption(meta)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["reply"],
                    "citations": data["citations"],
                    "cache_hit": data["cache_hit"],
                    "latency": data["latency_ms"]
                })
            else:
                st.error(f"Backend error {res.status_code}")
                st.code(res.text)