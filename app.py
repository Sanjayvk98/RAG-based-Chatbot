from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
#from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import streamlit as st
import os
import tempfile
import uuid
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    documents: list
    answer: str
    response: Literal["PDF", "general"]
    chat_history: list[str]

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
        response = model.invoke(formatted_prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        answer_clean = answer.strip().upper()
        classification = "PDF" if "PDF" in answer_clean else ("general" if "GENERAL" in answer_clean else "general")
        return {**state, "response": classification}
    except Exception:
        return {**state, "response": "general"}

def get_pdf_text(pdf_docs):
    """Load PDF files using LangChain's PyPDFLoader and return combined text."""
    text = ""
    if pdf_docs is None:
        return text
    for pdf in pdf_docs:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf.getvalue())
                tmp_path = tmp_file.name
            
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            
            for doc in documents:
                text += doc.page_content + "\n"
            
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
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
        st.error(f"Error retrieving documents: {str(e)}")
        return {
            **state,
            "documents": []
        }

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
        st.error(error_msg)
        return {
            **state,
            "answer": error_msg,
            "messages": state["messages"] + [AIMessage(content=error_msg)]
        }

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
        st.error(error_msg)
        return state

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

# Cache the checkpointer
@st.cache_resource
def get_checkpointer():
    """Get or create the checkpointer (cached)."""
    return InMemorySaver()

def create_rag_graph():
    """Create and compile the LangGraph with classification routing."""
    workflow = StateGraph(GraphState)
    
    # Add all nodes
    workflow.add_node("classify", classify_question)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("refine", refine_answer)
    workflow.add_node("general_answer", generate_general_answer)
    
    # Set entry point
    workflow.set_entry_point("classify")
    
    # Add conditional routing after classification
    workflow.add_conditional_edges(
        "classify",
        should_retrieve,
        {
            "pdf_flow": "retrieve",
            "general_flow": "general_answer"
        }
    )
    
    # PDF flow: retrieve -> generate -> refine -> END
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "refine")
    workflow.add_edge("refine", END)
    
    # General flow: general_answer -> END
    workflow.add_edge("general_answer", END)
    
    # Use the cached checkpointer
    checkpointer = get_checkpointer()
    app = workflow.compile(checkpointer=checkpointer)
    return app

# Cache the graph
@st.cache_resource
def get_rag_graph():
    """Get or create the RAG graph (cached)."""
    return create_rag_graph()

def user_input(user_question):
    """Process user question using LangGraph."""
    try:
        # Ensure a stable thread id for this session
        if "thread_id" not in st.session_state:
            st.session_state.thread_id = f"thread-{uuid.uuid4()}"

        # Get the cached graph
        graph = get_rag_graph()
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        # Build message history from session state (exclude current message)
        message_history = []
        for m in st.session_state.messages[:-1]:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "user":
                message_history.append(HumanMessage(content=content))
            elif role == "assistant":
                message_history.append(AIMessage(content=content))

        # Build chat history as strings (for context) - last 10 messages
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
    # Generate a new thread ID to start fresh
    st.session_state.thread_id = f"thread-{uuid.uuid4()}"

def main():
    st.set_page_config(
        page_title="DeepSeek PDF Chatbot (LangGraph)",
        page_icon="🤖"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
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
                        st.error("No text could be extracted from the PDF files. Please check if the files are valid PDFs.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Done")

    # Main content area for displaying chat messages
    st.title("Chat with PDF files using DeepSeek🤖 (LangGraph)")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
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

    # Display chat messages and bot response
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

if __name__ == "__main__":
    main()