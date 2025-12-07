# RAG-based-Chatbot

Overview
A RAG (Retrieval-Augmented Generation) chatbot built with Streamlit and LangGraph. It answers questions from uploaded PDFs using DeepSeek-V3.2-Exp via HuggingFace.
Architecture
Uses LangGraph to orchestrate a multi-step workflow: document retrieval → answer generation → answer refinement.
Components
1. State management (GraphState)
messages: conversation history
question: current user question
documents: retrieved document chunks
answer: generated/refined answer
2. PDF processing
get_pdf_text(): extracts text from uploaded PDFs using PyPDFLoader
get_text_chunks(): splits text into 10,000-character chunks with 1,000-character overlap
get_vector_store(): creates embeddings (sentence-transformers/all-MiniLM-L6-v2) and stores them in ChromaDB
3. LangGraph workflow nodes
retrieve_documents(): semantic search in ChromaDB (top 4 chunks)
generate_answer(): generates an initial answer from retrieved context using DeepSeek
refine_answer(): refines the answer for clarity and structure
4. Graph orchestration
Flow: retrieve → generate → refine → END
Uses StateGraph to manage state across nodes
Graph is cached with @st.cache_resource
5. Streamlit UI
Sidebar: PDF upload and processing
Main area: chat interface with message history
Features: clear chat history, real-time responses, error handling
Workflow
User uploads PDFs → text extraction → chunking → vectorization → stored in ChromaDB
User asks a question → graph execution:
Retrieves relevant chunks
Generates initial answer
Refines the answer
Returns the refined answer
Technologies
LangChain: document processing, embeddings, vector store
LangGraph: workflow orchestration
Streamlit: web interface
ChromaDB: vector database
HuggingFace: embeddings and DeepSeek LLM
DeepSeek-V3.2-Exp: language model
Features
Multi-PDF support
Persistent vector storage (ChromaDB)
Two-stage generation (generate + refine)
Conversation history
Error handling with user feedback
Semantic search for relevant context
Configuration
Requires HUGGINGFACEHUB_API_TOKEN or HF_TOKEN in .env
Vector store persisted in chroma_db/
Uses CPU for embeddings
This setup provides a structured RAG pipeline with answer refinement for clearer, more organized responses.
