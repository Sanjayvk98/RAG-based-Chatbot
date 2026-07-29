"""
Azure OpenAI RAG Chatbot with Document Intelligence & BM25 Hybrid Search
- Removes LLM-as-judge classification
- Uses Azure Document Intelligence for smart PDF extraction
- Hybrid retrieval: BM25 (keyword) + semantic (vector)
- Azure OpenAI for LLM and embeddings
"""

import os
import streamlit as st
import uuid
import tempfile
import json
from typing import TypedDict, Annotated, Literal
from pathlib import Path

# LangChain imports
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

# BM25 + Azure Document Intelligence
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()

# ==================== AZURE CONFIG ====================
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small")

# Document Intelligence (Form Recognizer)
DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
DOCUMENT_INTELLIGENCE_KEY = os.getenv("DOCUMENT_INTELLIGENCE_API_KEY")

# ==================== INITIALIZE AZURE CLIENTS ====================
# ==================== INITIALIZE AZURE CLIENTS ====================
def init_azure_clients():
    """Initialize Azure OpenAI and Document Intelligence clients with strict variable targeting."""
    try:
        # Load environment variables cleanly, stripping spaces or trailing slashes
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        
        # FORCE targeting the exact key used in your working test script
        embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME", "").strip()
        llm_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4").strip()

        # Fail early locally if your env file didn't load properly
        if not embedding_deployment:
            st.error("🚨 AZURE_EMBEDDING_DEPLOYMENT_NAME is completely blank or missing in your .env file!")
            raise ValueError("Missing embedding deployment configuration")

        llm = AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-06-01",  # Fixed to your working stable version
            azure_deployment=llm_deployment,
            temperature=0.2,
            max_tokens=1000
        )
        
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-06-01",  # Fixed to your working stable version
            azure_deployment=embedding_deployment  # Strictly uses your custom deployment slot name
        )
        
        doc_client = None
        if DOCUMENT_INTELLIGENCE_ENDPOINT and DOCUMENT_INTELLIGENCE_KEY:
            doc_client = DocumentIntelligenceClient(
                endpoint=DOCUMENT_INTELLIGENCE_ENDPOINT,
                credential=AzureKeyCredential(DOCUMENT_INTELLIGENCE_KEY)
            )
        
        return llm, embeddings, doc_client
    except Exception as e:
        st.error(f"Azure initialization failed: {str(e)}")
        raise



# ==================== DOCUMENT INTELLIGENCE PDF EXTRACTION ====================
def extract_pdf_with_document_intelligence(pdf_path: str, doc_client):
    """Extract text from PDF using Azure Document Intelligence (Form Recognizer)."""
    try:
        with open(pdf_path, "rb") as pdf_file:
            poller = doc_client.begin_analyze_document(
                "prebuilt-document",
                pdf_file,
                locale="en-US"
            )
            result = poller.result()
        
        # Extract text maintaining structure
        text = ""
        if result.content:
            text = result.content
        
        return text
    except Exception as e:
        st.warning(f"Document Intelligence failed, falling back to PyPDFLoader: {str(e)}")
        return None

def extract_pdf_text(pdf_docs, doc_client=None):
    """Extract text from PDFs using Document Intelligence or PyPDFLoader."""
    all_text = ""
    
    if pdf_docs is None or len(pdf_docs) == 0:
        return all_text
    
    for pdf in pdf_docs:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf.getvalue())
                tmp_path = tmp_file.name
            
            # Try Document Intelligence first
            text = ""
            if doc_client:
                text = extract_pdf_with_document_intelligence(tmp_path, doc_client)
            
            # Fallback to PyPDFLoader
            if not text:
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                text = "\n".join([doc.page_content for doc in documents])
            
            all_text += text + "\n"
            
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
    
    return all_text

# ==================== BM25 HYBRID SEARCH ====================
class HybridRetriever:
    """Combines BM25 keyword search with semantic vector search."""
    
    def __init__(self, chunks: list, embeddings, top_k: int = 4):
        self.chunks = chunks
        self.embeddings = embeddings
        self.top_k = top_k
        
        # Initialize BM25
        self.bm25 = BM25Okapi([chunk.split() for chunk in chunks])
        
        # Initialize vector store
        documents = [Document(page_content=chunk) for chunk in chunks]
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="chroma_db"
        )
    
    def retrieve(self, query: str) -> list:
        """Retrieve using both BM25 and semantic search, then combine."""
        # BM25 retrieval
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:self.top_k]
        
        # Semantic retrieval
        semantic_docs = self.vector_store.similarity_search(query, k=self.top_k)
        
        # Combine results (deduplicate by content)
        seen_content = set()
        combined_docs = []
        
        # Add BM25 results first (higher weight on exact keywords)
        for idx in bm25_top_indices[:self.top_k]:
            content = self.chunks[idx]
            if content not in seen_content:
                combined_docs.append(Document(page_content=content))
                seen_content.add(content)
        
        # Add semantic results
        for doc in semantic_docs:
            if doc.page_content not in seen_content:
                combined_docs.append(doc)
                seen_content.add(doc.page_content)
        
        return combined_docs[:self.top_k]

# ==================== LANGGRAPH STATE ====================
class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    documents: list
    answer: str
    chat_history: list[str]

# ==================== LANGGRAPH NODES ====================
def retrieve_documents(state: GraphState, retriever: HybridRetriever) -> GraphState:
    """Retrieve relevant documents using hybrid BM25 + semantic search."""
    try:
        docs = retriever.retrieve(state["question"])
        return {**state, "documents": docs}
    except Exception as e:
        st.error(f"Retrieval error: {str(e)}")
        return {**state, "documents": []}

def generate_answer(state: GraphState, llm: AzureChatOpenAI) -> GraphState:
    """Generate answer using retrieved documents."""
    try:
        if not state["documents"]:
            answer = "I don't have relevant information in the uploaded documents to answer this question."
            return {
                **state,
                "answer": answer,
                "messages": state["messages"] + [AIMessage(content=answer)]
            }
        
        # Build context
        context = "\n\n".join([
            f"[Source {i+1}]\n{doc.page_content[:500]}"
            for i, doc in enumerate(state["documents"])
        ])
        
        # Build chat history
        history_text = "\n".join(state.get("chat_history", [])[-5:])  # Last 5 messages
        
        # Create prompt
        prompt_template = """You are a helpful assistant answering questions based on provided documents.

INSTRUCTIONS:
- Only use information from the provided context
- If the answer is not in the context, say "I don't have that information"
- Be concise and direct
- Cite which source (1-4) you're referencing when relevant

Previous conversation:
{chat_history}

Context from documents:
{context}

Question: {question}

Answer:"""
        
        messages = [
            {"role": "system", "content": "You are a helpful document Q&A assistant. Answer based only on provided context."},
            {"role": "user", "content": prompt_template.format(
                context=context,
                question=state["question"],
                chat_history=history_text
            )}
        ]
        
        response = llm.invoke(messages)
        answer = response.content
        
        return {
            **state,
            "answer": answer,
            "messages": state["messages"] + [AIMessage(content=answer)]
        }
    except Exception as e:
        error_msg = f"Answer generation error: {str(e)}"
        st.error(error_msg)
        return {
            **state,
            "answer": error_msg,
            "messages": state["messages"] + [AIMessage(content=error_msg)]
        }

# ==================== BUILD GRAPH ====================
def create_rag_graph(retriever: HybridRetriever, llm: AzureChatOpenAI):
    """Create simplified LangGraph without classification."""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("retrieve", lambda state: retrieve_documents(state, retriever))
    workflow.add_node("generate", lambda state: generate_answer(state, llm))
    
    # Define flow
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    # Compile with checkpointer
    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)

# ==================== STREAMLIT UI ====================
@st.cache_resource
def get_retriever_and_graph(embeddings):
    """Initialize retriever and graph (cached)."""
    # Placeholder - will be set after PDF processing
    return None

def process_pdfs(pdf_docs, embeddings, llm, doc_client):
    """Process PDFs and create retriever."""
    with st.spinner("Extracting text from PDFs..."):
        # Extract text
        raw_text = extract_pdf_text(pdf_docs, doc_client)
        
        if not raw_text.strip():
            st.error("No text extracted from PDFs")
            return None
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(raw_text)
        
        # Create hybrid retriever
        retriever = HybridRetriever(chunks, embeddings, top_k=4)
        
        # Create graph
        graph = create_rag_graph(retriever, llm)
        
        st.success(f"✅ Processed {len(chunks)} chunks from PDFs")
        return {"retriever": retriever, "graph": graph}

def main():
    st.set_page_config(
        page_title="Azure OpenAI RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Azure OpenAI RAG Chatbot")
    st.markdown("*Powered by Azure OpenAI, Document Intelligence & BM25 Hybrid Search*")
    
    # Initialize Azure clients
    try:
        llm, embeddings, doc_client = init_azure_clients()
    except Exception as e:
        st.error(f"Failed to initialize Azure clients. Check your .env file: {str(e)}")
        return
    
    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_context" not in st.session_state:
        st.session_state.rag_context = None
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"thread-{uuid.uuid4()}"
    
    # Sidebar
    with st.sidebar:
        st.header("📄 Document Upload")
        pdf_docs = st.file_uploader(
            "Upload PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDFs to ask questions about"
        )
        
        if st.button("🔄 Process PDFs", use_container_width=True):
            if not pdf_docs:
                st.warning("Please upload at least one PDF")
            else:
                result = process_pdfs(pdf_docs, embeddings, llm, doc_client)
                if result:
                    st.session_state.rag_context = result
                    st.session_state.messages = []
                    st.session_state.thread_id = f"thread-{uuid.uuid4()}"
        
        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = f"thread-{uuid.uuid4()}"
        
        st.divider()
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Features:**
        - Azure OpenAI for responses
        - Document Intelligence for PDF extraction
        - BM25 + Semantic hybrid search
        - Chat history awareness
        """)
    
    # Main chat area
    if not st.session_state.rag_context:
        st.info("👈 Upload PDFs in the sidebar to get started")
        return
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if user_input := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Process with RAG
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    retriever = st.session_state.rag_context["retriever"]
                    graph = st.session_state.rag_context["graph"]
                    
                    # Build message history
                    message_history = []
                    for m in st.session_state.messages[:-1]:
                        if m["role"] == "user":
                            message_history.append(HumanMessage(content=m["content"]))
                        else:
                            message_history.append(AIMessage(content=m["content"]))
                    
                    # Build chat history strings
                    chat_history = [f"{m['role']}: {m['content']}" for m in st.session_state.messages[-10:]]
                    
                    # Invoke graph
                    initial_state = {
                        "messages": message_history + [HumanMessage(content=user_input)],
                        "question": user_input,
                        "documents": [],
                        "answer": "",
                        "chat_history": chat_history,
                    }
                    
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    final_state = graph.invoke(initial_state, config=config)
                    
                    answer = final_state.get("answer", "No response generated")
                    st.markdown(answer)
                    
                    # Add assistant message
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = f"⚠️ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()