import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# CHANGE THESE TWO LINES:
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

# Load the API key from the .env file
load_dotenv()

def get_answer(question, persist_directory="faiss_index"):
    # 1. Load the exact same embedding model used in ingestion
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 2. Load the Vector Database
    try:
        # allow_dangerous_deserialization is required to load local FAISS files safely
        vector_db = FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return "⚠️ Error: Database not found. Please upload and process a PDF first."

    # 3. Initialize the Deep AI LLM (Updated to active Llama 3.1 model)
    llm = ChatGroq(
        temperature=0, # 0 means strictly factual (no hallucinations)
        model_name="llama-3.1-8b-instant" 
    )

    # 4. Create the System Prompt
    system_prompt = (
        "You are an intelligent enterprise assistant. Use the following pieces of retrieved "
        "context to answer the user's question. If the answer is not in the context, just say "
        "that you don't know. Do not make up information. Keep the answer professional.\n\n"
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 5. Build the RAG Chain (Retrieve -> Prompt -> Generate)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3}) # Get top 3 most relevant chunks
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # 6. Execute and return the answer
    response = rag_chain.invoke({"input": question})
    return response["answer"]