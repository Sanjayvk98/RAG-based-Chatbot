
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
import os
import tempfile
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


load_dotenv()

# Define the state for LangGraph
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    documents: list
    answer: str

# read all pdf files and return documents
def get_pdf_text(pdf_docs):
    """Load PDF files using LangChain's PyPDFLoader and return combined text."""
    text = ""
    if pdf_docs is None:
        return text
    for pdf in pdf_docs:
        tmp_path = None
        try:
            # Save uploaded file temporarily to use with PyPDFLoader
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf.getvalue())
                tmp_path = tmp_file.name
            
            # Use PyPDFLoader to load the PDF
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            
            # Combine all pages from this PDF
            for doc in documents:
                text += doc.page_content + "\n"
            
            # Clean up temporary file
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
            # Clean up temp file if it exists
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
    return text

# split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# get embeddings for each chunk
def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    # Convert string chunks to Document objects
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})
    return retriever

# LangGraph node: Retrieve documents
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

# LangGraph node: Generate answer
def generate_answer(state: GraphState) -> GraphState:
    """Generate answer using the retrieved documents."""
    try:
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}\n
        Question: \n{question}\n

        Answer:
        """

        model = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id="deepseek-ai/DeepSeek-V3.2-Exp",
                task="text-generation",
                huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
            ),
            temperature=0.3
        )
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Format context from documents
        context = "\n\n".join([doc.page_content for doc in state["documents"]])
        
        # Format prompt
        formatted_prompt = prompt.format(
            context=context,
            question=state["question"]
        )
        
        # Generate response
        response = model.invoke(formatted_prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        
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

# LangGraph node: Refine answer for better structure
def refine_answer(state: GraphState) -> GraphState:
    """Refine the generated answer to make it more structured and polished."""
    try:
        refinement_prompt_template = """
        You are a helpful assistant that refines and structures answers to make them clearer and more organized.
        
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
            temperature=0.2  # Lower temperature for more consistent refinement
        )
        
        prompt = PromptTemplate(
            template=refinement_prompt_template,
            input_variables=["question", "initial_answer"]
        )
        
        # Format prompt with the initial answer
        formatted_prompt = prompt.format(
            question=state["question"],
            initial_answer=state["answer"]
        )
        
        # Generate refined response
        response = model.invoke(formatted_prompt)
        refined_answer = response.content if hasattr(response, 'content') else str(response)
        
        return {
            **state,
            "answer": refined_answer,  # Update answer with refined version
            "messages": state["messages"][:-1] + [AIMessage(content=refined_answer)]  # Replace last message with refined answer
        }
    except Exception as e:
        error_msg = f"Error refining answer: {str(e)}"
        st.error(error_msg)
        # If refinement fails, keep the original answer
        return {
            **state,
            "messages": state["messages"]  # Keep original answer
        }

# Build the LangGraph
def create_rag_graph():
    """Create and compile the LangGraph for RAG."""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("refine", refine_answer)
    
    # Set entry point
    workflow.set_entry_point("retrieve")
    
    # Add edges: retrieve -> generate -> refine -> END
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "refine")
    workflow.add_edge("refine", END)

    # Compile the graph
    app = workflow.compile()
    return app

# Initialize graph (will be cached in Streamlit)
@st.cache_resource
def get_rag_graph():
    """Get or create the RAG graph (cached)."""
    return create_rag_graph()

def user_input(user_question):
    """Process user question using LangGraph."""
    try:
        # Get the graph
        graph = get_rag_graph()
        
        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=user_question)],
            "question": user_question,
            "documents": [],
            "answer": ""
        }
        
        # Run the graph
        final_state = graph.invoke(initial_state)
        
        answer = final_state.get("answer", "")
        print(answer)
        return {"output_text": answer}
    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        st.error(error_msg)
        return {"output_text": error_msg}

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}]

def main():
    st.set_page_config(
        page_title="DeepSeek PDF Chatbot (LangGraph)",
        page_icon="🤖"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
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
            {"role": "assistant", "content": "upload some pdfs and ask me a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = response.get('output_text', str(response))
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)

if __name__ == "__main__":
    main()
