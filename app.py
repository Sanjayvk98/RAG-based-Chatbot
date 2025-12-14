from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tracers.context import tracing_v2_enabled
import streamlit as st
import os
import tempfile
import uuid
from typing import TypedDict, Annotated, Literal, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
from langsmith import traceable

load_dotenv()

# ============================================================================
# LANGSMITH CONFIGURATION
# ============================================================================
# IMPORTANT: Load API key from environment (use LANGCHAIN_API_KEY not LANGSMITH_API_KEY)
langchain_api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY", "")

# Verify key is loaded
if langchain_api_key:
    print(f"✅ LangSmith API Key loaded (starts with: {langchain_api_key[:10]}...)")
else:
    print("⚠️  WARNING: LANGCHAIN_API_KEY not found in environment!")
    print("   Set it in your .env file: LANGCHAIN_API_KEY=lsv2_pt_xxxxx")

# Set up LangSmith environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "pdf-rag-chatbot")

# Verify LangSmith connection
try:
    from langsmith import Client
    client = Client()
    print(f"✅ LangSmith connected! Project: {os.environ['LANGCHAIN_PROJECT']}")
    print(f"   View traces at: https://smith.langchain.com/")
except Exception as e:
    print(f"❌ LangSmith connection failed: {e}")
    print("   Tracing will be disabled.")

# ============================================================================
# FASTAPI MODELS
# ============================================================================
class ChatRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None
    chat_history: Optional[List[dict]] = []

class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    classification: str

class ProcessPDFResponse(BaseModel):
    message: str
    chunks_created: int

# ============================================================================
# GRAPH STATE
# ============================================================================
class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    documents: list
    answer: str
    response: Literal["PDF", "general"]
    chat_history: list[str]

# ============================================================================
# GRAPH NODES
# ============================================================================



@traceable
def classify_question(state: GraphState) -> GraphState:
    """Classify the question as either PDF-related or general."""
    try:
        prompt_template = """
        You are a Classifier that classifies questions into one of two categories: "PDF" or "general".
        
        Rules:
        - Return "PDF" if the question is about content from uploaded PDF documents or requires information from PDFs
        - Return "general" if the question is a general knowledge question, greeting, or doesn't require PDF content
        
        The question is: {question}
        
        Respond with ONLY one word: either "PDF" or "general"
        Response:
        """
        model = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id="deepseek-ai/DeepSeek-V3.2-Exp",
                task="text-generation",
                huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
            ),
            temperature=0.3
        )
        prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
        formatted_prompt = prompt.format(question=state["question"])
        
        # LangSmith will automatically trace this invocation
        response = model.invoke(formatted_prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        answer_clean = answer.strip().upper()
        classification = "PDF" if "PDF" in answer_clean else ("general" if "GENERAL" in answer_clean else "general")
        return {**state, "response": classification}
    except Exception as e:
        print(f"Classification error: {e}")
        return {**state, "response": "general"}

@traceable
def retrieve_documents(state: GraphState) -> GraphState:
    """Retrieve relevant documents from the vector store."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        
        new_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
        question = state["question"]
        docs = new_db.similarity_search(question, k=4)
        
        return {
            **state,
            "documents": docs
        }
    except Exception as e:
        print(f"Error retrieving documents: {str(e)}")
        return {
            **state,
            "documents": []
        }

@traceable
def generate_answer(state: GraphState) -> GraphState:
    """Generate answer using the retrieved documents."""
    try:
        prompt_template = """
        You are a grounded assistant. Use ONLY:
        1) Retrieved context
        2) Recent chat history

        If the answer is not in those, reply exactly:
        "I don't have that in the PDFs or prior conversation."

        Chat history:
        {chat_history}

        Context:
        {context}

        Question:
        {question}

        Answer concisely, cite which context snippets you used.
        """

        model = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id="deepseek-ai/DeepSeek-V3.2-Exp",
                task="text-generation",
                huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
            ),
            temperature=0.15
        )

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question", "chat_history"]
        )

        if not state["documents"]:
            refusal = "I don't have that in the PDFs or prior conversation."
            return {
                **state,
                "answer": refusal,
                "messages": state["messages"] + [AIMessage(content=refusal)]
            }

        context = "\n\n".join([doc.page_content for doc in state["documents"]])
        history_text = "\n".join(state.get("chat_history", []))

        formatted_prompt = prompt.format(
            context=context,
            question=state["question"],
            chat_history=history_text,
        )

        response = model.invoke(formatted_prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        return {
            **state,
            "answer": answer,
            "messages": state["messages"] + [AIMessage(content=answer)]
        }
    except Exception as e:
        error_msg = f"Error generating answer: {str(e)}"
        print(error_msg)
        return {
            **state,
            "answer": error_msg,
            "messages": state["messages"] + [AIMessage(content=error_msg)]
        }

@traceable
def refine_answer(state: GraphState) -> GraphState:
    """Refine the generated answer to make it more structured and polished."""
    try:
        refinement_prompt_template = """
        You are a helpful assistant that refines and structures answers to make them clearer and more organized.
        
        Chat history:
        {chat_history}

        Original Question: {question}
        Initial Answer: {initial_answer}

        Please refine this answer to:
        1. Make it more structured and well-organized
        2. Ensure it's clear and easy to read
        3. Maintain all the important information from the original answer
        4. Improve the flow and coherence
        5. If the answer mentions "answer is not available in the context", keep that message but make it more polite

        Provide the refined answer:
        """

        model = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id="deepseek-ai/DeepSeek-V3.2-Exp",
                task="text-generation",
                huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
            ),
            temperature=0.2
        )
        
        prompt = PromptTemplate(
            template=refinement_prompt_template,
            input_variables=["question", "initial_answer", "chat_history"]
        )

        history_text = "\n".join(state.get("chat_history", []))
        formatted_prompt = prompt.format(
            question=state["question"],
            initial_answer=state["answer"],
            chat_history=history_text,
        )

        response = model.invoke(formatted_prompt)
        refined_answer = response.content if hasattr(response, 'content') else str(response)

        return {
            **state,
            "answer": refined_answer,
            "messages": state["messages"][:-1] + [AIMessage(content=refined_answer)]
        }
    except Exception as e:
        error_msg = f"Error refining answer: {str(e)}"
        print(error_msg)
        return state

@traceable
def generate_general_answer(state: GraphState) -> GraphState:
    """Generate answer for general questions without PDF context."""
    try:
        prompt_template = """
        You are a helpful AI assistant. Answer the following question based on your general knowledge.
        
        Chat history (for context):
        {chat_history}
        
        Question: {question}
        
        Provide a clear, detailed, and helpful answer:
        """

        model = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id="deepseek-ai/DeepSeek-V3.2-Exp",
                task="text-generation",
                huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
            ),
            temperature=0.7
        )
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["question", "chat_history"]
        )
        
        history_text = "\n".join(state.get("chat_history", []))
        formatted_prompt = prompt.format(
            question=state["question"],
            chat_history=history_text
        )
        response = model.invoke(formatted_prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return {
            **state,
            "answer": answer,
            "messages": state["messages"] + [AIMessage(content=answer)]
        }
        
    except Exception as e:
        error_msg = f"Error generating general answer: {str(e)}"
        return {
            **state,
            "answer": error_msg,
            "messages": state["messages"] + [AIMessage(content=error_msg)]
        }

def should_retrieve(state: GraphState) -> Literal["pdf_flow", "general_flow"]:
    """Route based on classification."""
    return "pdf_flow" if state.get("response") == "PDF" else "general_flow"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
@traceable
def get_pdf_text(pdf_files):
    """Load PDF files using LangChain's PyPDFLoader and return combined text."""
    text = ""
    if pdf_files is None:
        return text
    for pdf in pdf_files:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf.getvalue() if hasattr(pdf, 'getvalue') else pdf.file.read())
                tmp_path = tmp_file.name
            
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            
            for doc in documents:
                text += doc.page_content + "\n"
            
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception as e:
            print(f"Error reading PDF: {str(e)}")
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=50)
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})
    return retriever

# ============================================================================
# GRAPH CREATION
# ============================================================================
# Global checkpointer for both Streamlit and FastAPI
checkpointer = InMemorySaver()

@traceable
def create_rag_graph():
    """Create and compile the LangGraph with classification routing."""
    workflow = StateGraph(GraphState)
    
    workflow.add_node("classify", classify_question)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("refine", refine_answer)
    workflow.add_node("general_answer", generate_general_answer)
    
    workflow.set_entry_point("classify")
    
    workflow.add_conditional_edges(
        "classify",
        should_retrieve,
        {
            "pdf_flow": "retrieve",
            "general_flow": "general_answer"
        }
    )
    
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "refine")
    workflow.add_edge("refine", END)
    workflow.add_edge("general_answer", END)
    
    app = workflow.compile(checkpointer=checkpointer)
    return app

# Cache the graph
@st.cache_resource
def get_rag_graph():
    """Get or create the RAG graph (cached for Streamlit)."""
    return create_rag_graph()

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================
# Global graph instance for FastAPI
fastapi_graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize graph on startup."""
    global fastapi_graph
    fastapi_graph = create_rag_graph()
    print("FastAPI: RAG Graph initialized with LangSmith tracing")
    yield
    # Cleanup if needed
    print("FastAPI: Shutting down")

app = FastAPI(
    title="PDF RAG Chatbot API",
    description="API for chatting with PDF documents using RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "PDF RAG Chatbot API",
        "endpoints": {
            "POST /upload-pdfs": "Upload and process PDF files",
            "POST /chat": "Send a question to the chatbot",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "langsmith_enabled": os.getenv("LANGCHAIN_TRACING_V2") == "true"}

@app.post("/upload-pdfs", response_model=ProcessPDFResponse)
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """
    Upload and process PDF files.
    Creates vector embeddings and stores them in ChromaDB.
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        # Convert UploadFile to format compatible with get_pdf_text
        pdf_contents = []
        for file in files:
            content = await file.read()
            # Create a simple wrapper object
            class PDFWrapper:
                def __init__(self, content):
                    self._content = content
                def getvalue(self):
                    return self._content
            pdf_contents.append(PDFWrapper(content))
        
        # Process PDFs with LangSmith tracing
        with tracing_v2_enabled(project_name=os.getenv("LANGCHAIN_PROJECT", "pdf-rag-chatbot")):
            raw_text = get_pdf_text(pdf_contents)
            
            if not raw_text.strip():
                raise HTTPException(status_code=400, detail="No text could be extracted from PDFs")
            
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
        
        return ProcessPDFResponse(
            message="PDFs processed successfully",
            chunks_created=len(text_chunks)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDFs: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a question to the chatbot.
    Uses LangGraph with LangSmith tracing.
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Generate thread_id if not provided
        thread_id = request.thread_id or f"thread-{uuid.uuid4()}"
        config = {"configurable": {"thread_id": thread_id}}
        
        # Build message history
        message_history = []
        for msg in request.chat_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                message_history.append(HumanMessage(content=content))
            elif role == "assistant":
                message_history.append(AIMessage(content=content))
        
        # Build chat history strings
        chat_history_strings = [
            f"{msg.get('role')}: {msg.get('content')}"
            for msg in request.chat_history[-10:]
            if msg.get('role') in ["user", "assistant"]
        ]
        
        initial_state = {
            "messages": message_history + [HumanMessage(content=request.question)],
            "question": request.question,
            "documents": [],
            "answer": "",
            "response": "general",
            "chat_history": chat_history_strings,
        }
        
        # Invoke graph with LangSmith tracing
        with tracing_v2_enabled(project_name=os.getenv("LANGCHAIN_PROJECT", "pdf-rag-chatbot")):
            final_state = fastapi_graph.invoke(initial_state, config=config)
        
        answer = final_state.get("answer", "")
        classification = final_state.get("response", "general")
        
        return ChatResponse(
            answer=answer,
            thread_id=thread_id,
            classification=classification
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================
def user_input(user_question):
    """Process user question using LangGraph with LangSmith tracing."""
    try:
        if "thread_id" not in st.session_state:
            st.session_state.thread_id = f"thread-{uuid.uuid4()}"

        graph = get_rag_graph()
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        message_history = []
        for m in st.session_state.messages[:-1]:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "user":
                message_history.append(HumanMessage(content=content))
            elif role == "assistant":
                message_history.append(AIMessage(content=content))

        chat_history_strings = []
        for m in st.session_state.messages[-10:]:
            role = m.get("role", "")
            content = m.get("content", "")
            if role in ["user", "assistant"]:
                chat_history_strings.append(f"{role}: {content}")

        initial_state = {
            "messages": message_history + [HumanMessage(content=user_question)],
            "question": user_question,
            "documents": [],
            "answer": "",
            "response": "general",
            "chat_history": chat_history_strings,
        }

        # Use LangSmith tracing context
        with tracing_v2_enabled(project_name=os.getenv("LANGCHAIN_PROJECT", "pdf-rag-chatbot")):
            final_state = graph.invoke(initial_state, config=config)
        
        answer = final_state.get("answer", "")
        return {"output_text": answer}
    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        st.error(error_msg)
        return {"output_text": error_msg}

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}
    ]
    st.session_state.thread_id = f"thread-{uuid.uuid4()}"

def main():
    st.set_page_config(
        page_title="DeepSeek PDF Chatbot (LangGraph + LangSmith)",
        page_icon="🤖"
    )

    # Display LangSmith status
    langsmith_enabled = os.getenv("LANGCHAIN_TRACING_V2") == "true"
    
    with st.sidebar:
        st.title("Menu:")
        
        # LangSmith status indicator
        if langsmith_enabled:
            st.success("🟢 LangSmith Tracing: Enabled")
            st.caption(f"Project: {os.getenv('LANGCHAIN_PROJECT', 'pdf-rag-chatbot')}")
        else:
            st.warning("🟡 LangSmith Tracing: Disabled")
        
        st.divider()
        
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", 
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            if pdf_docs is None or len(pdf_docs) == 0:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No text could be extracted from the PDF files.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success(f"Done! Created {len(text_chunks)} chunks")

    st.title("Chat with PDF files using DeepSeek🤖")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "upload some pdfs and ask me a question"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
        last_msg = st.session_state.messages[-1]
        user_question = last_msg.get("content") or ""
        if not user_question:
            st.warning("Your last message was empty; please enter a question.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = user_input(user_question)
                    placeholder = st.empty()
                    full_response = response.get('output_text', str(response))
                    placeholder.markdown(full_response)
            if response is not None:
                message = {"role": "assistant", "content": full_response}
                st.session_state.messages.append(message)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        # Run FastAPI
        print("Starting FastAPI server...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Run Streamlit
        main()