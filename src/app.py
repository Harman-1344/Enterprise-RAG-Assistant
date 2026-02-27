import streamlit as st
import os
from ingestion import create_vector_db
from rag_pipeline import get_answer

# --- UI Configuration ---
st.set_page_config(page_title="Enterprise AI Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Enterprise RAG Assistant")
st.markdown("Upload a PDF document and ask questions about it using Deep AI vector search.")

# Ensure data directory exists for temporary file storage
os.makedirs("data", exist_ok=True)

# --- Sidebar: File Upload & Processing ---
with st.sidebar:
    st.header("📄 1. Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file is not None:
        # Save file to the /data folder
        temp_pdf_path = os.path.join("data", uploaded_file.name)
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded: {uploaded_file.name}")
        
        # Process Button
        if st.button("⚙️ Process Document"):
            with st.spinner("Chunking text and generating Deep AI Embeddings..."):
                create_vector_db(temp_pdf_path)
                st.success("Database built! You can now ask questions.")

# --- Main Page: Chat Interface ---
st.header("💬 2. Ask Questions")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
if prompt := st.chat_input("Ask a question about the uploaded document..."):
    
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate and display AI response
    with st.chat_message("assistant"):
        with st.spinner("Searching vector database and generating answer..."):
            response = get_answer(prompt)
            st.markdown(response)
    
    # 3. Save AI response to history
    st.session_state.messages.append({"role": "assistant", "content": response})