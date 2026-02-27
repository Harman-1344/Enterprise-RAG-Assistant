# Enterprise RAG Assistant

## About

Welcome to the **Enterprise RAG Assistant** repository! This project is an advanced **Retrieval-Augmented Generation (RAG)** system that allows users to have natural language conversations with their private, complex PDF documents.

I built this project to solve a critical enterprise problem: extracting precise, context-aware information from large internal manuals, reports, or research papers without suffering from AI hallucinations. Instead of relying on traditional keyword matching (like TF-IDF), this system leverages Deep Learning to understand the true semantic meaning of both the document and the user's questions.

This repository showcases my ability to design, build, and deploy modern AI architectures. By integrating LangChain, HuggingFace dense embeddings, a FAISS vector database, and the open-source Llama-3.1 model, I created a robust pipeline that strictly grounds all AI responses in the provided source documents.

You can test the live deployed application here:
**[Live Demo: Enterprise RAG Assistant](https://enterprise-rag-assistant1.streamlit.app/)**

---

## Table of Contents

* [About](#about)
* [Key Features](#key-features)
* [Architecture & Flow](#architecture--flow)
* [Tech Stack](#tech-stack)
* [Installation & Local Setup](#installation--local-setup)
* [Usage](#usage)

---

## Key Features

* **Deep Semantic Search:** Uses `sentence-transformers` to convert text into high-dimensional vectors, vastly improving retrieval accuracy over standard search methods.
* **Vector Database:** Utilizes **FAISS** (Facebook AI Similarity Search) for lightning-fast retrieval of relevant document chunks.
* **Hallucination-Free LLM:** Powered by **Llama-3.1 (via Groq API)** with strict prompting to ensure the AI only answers based on the uploaded context.
* **User-Friendly Interface:** Deployed as a full-stack web application using **Streamlit** for seamless document uploading and chatting.

---

## Architecture & Flow

1. **Document Ingestion:**
   Parses uploaded PDFs and intelligently chunks the text using `RecursiveCharacterTextSplitter` to maintain context windows without cutting sentences in half.

2. **Deep Vector Embeddings:**
   Converts text chunks into mathematical vectors using HuggingFace's `all-MiniLM-L6-v2` embedding model.

3. **Storage:**
   Stores embeddings locally in a FAISS index.

4. **Generative Synthesis:**
   Retrieves the top most relevant document chunks based on the user's query and passes them to the LLM to generate a factual, conversational answer.

---

## Tech Stack

* **Language:** Python 3.10
* **AI Framework:** LangChain (v1.0 Modular Architecture)
* **Embeddings:** HuggingFace (`sentence-transformers`)
* **Vector Store:** FAISS CPU
* **LLM Engine:** Groq API (`llama-3.1-8b-instant`)
* **Frontend UI & Hosting:** Streamlit / Streamlit Community Cloud

---

## Installation & Local Setup

If you want to run this project on your local machine, follow these steps:

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Harman-1344/Enterprise-RAG-Assistant.git
cd Enterprise-RAG-Assistant
```

### 2️⃣ Create and activate a virtual environment

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Set up Environment Variables

Create a `.env` file in the root directory and add your free Groq API key:

```env
GROQ_API_KEY="your_groq_api_key_here"
```

### 5️⃣ Launch the Application

```bash
streamlit run src/app.py
```

---

## Usage

1. Open the live link or run the app locally.
2. Upload any text-heavy PDF document using the sidebar drag-and-drop feature.
3. Click **"Process Document"** to build the AI's vector database.
4. Use the chat interface to ask specific, analytical questions about the document's contents.



