# PDF RAG Chatbot with LangSmith & FastAPI

Enhanced version of your PDF RAG chatbot with **LangSmith tracing** and **FastAPI integration**.

## 🆕 What's New

### 1. **LangSmith Integration**
- **Automatic tracing** of all LLM calls and graph executions
- Monitor classification, retrieval, generation, and refinement steps
- View detailed traces in LangSmith dashboard
- Track performance metrics and debug issues

### 2. **FastAPI Integration**
- RESTful API endpoints for programmatic access
- Upload PDFs via API
- Chat with documents programmatically
- CORS enabled for frontend integration

## 📋 Prerequisites

1. **HuggingFace API Token**: Get from [HuggingFace](https://huggingface.co/settings/tokens)
2. **LangSmith API Key**: Get from [LangSmith](https://smith.langchain.com/)

## 🚀 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxx
LANGSMITH_API_KEY=lsv2_pt_xxxxxxxxxxxxx
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=pdf-rag-chatbot
```

## 🎯 Usage

### Option 1: Run Streamlit UI (Original)

```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

**Features:**
- Upload PDFs via UI
- Chat interface
- LangSmith status indicator in sidebar
- All interactions traced automatically

### Option 2: Run FastAPI Server (New)

```bash
python app.py --api
```

Access at: `http://localhost:8000`

**API Documentation:** `http://localhost:8000/docs`

## 📡 API Endpoints

### 1. Health Check
```bash
GET http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "langsmith_enabled": true
}
```

### 2. Upload PDFs
```bash
POST http://localhost:8000/upload-pdfs
Content-Type: multipart/form-data

files: [pdf_file1.pdf, pdf_file2.pdf]
```

**Response:**
```json
{
  "message": "PDFs processed successfully",
  "chunks_created": 145
}
```

### 3. Chat
```bash
POST http://localhost:8000/chat
Content-Type: application/json

{
  "question": "What is the main topic of the document?",
  "thread_id": "thread-12345",  // optional
  "chat_history": [              // optional
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"}
  ]
}
```

**Response:**
```json
{
  "answer": "The main topic of the document is...",
  "thread_id": "thread-12345",
  "classification": "PDF"
}
```

## 🔍 Using LangSmith

### View Traces

1. Go to [smith.langchain.com](https://smith.langchain.com/)
2. Select your project (`pdf-rag-chatbot`)
3. View all traced runs with:
   - Input/output for each step
   - Latency metrics
   - Token usage
   - Error traces

### Trace Details

Each conversation trace includes:
- **Classification**: Question categorization (PDF vs general)
- **Retrieval**: Vector search results
- **Generation**: Initial answer generation
- **Refinement**: Answer polishing
- **General Answer**: For non-PDF questions

### Benefits

- **Debug issues**: See exactly where failures occur
- **Optimize performance**: Identify slow steps
- **Monitor quality**: Review model outputs
- **Track costs**: Monitor token usage

## 🧪 Example Usage

### Python Client

```python
import requests

# Upload PDF
with open('document.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/upload-pdfs',
        files={'files': f}
    )
print(response.json())

# Chat
response = requests.post(
    'http://localhost:8000/chat',
    json={
        'question': 'What are the key findings?',
        'thread_id': 'my-session-123'
    }
)
print(response.json()['answer'])
```

### cURL

```bash
# Upload PDF
curl -X POST "http://localhost:8000/upload-pdfs" \
  -F "files=@document.pdf"

# Chat
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Summarize the document",
    "thread_id": "session-1"
  }'
```

## 🏗️ Architecture

### LangGraph Flow

```
User Question
    ↓
[Classify] → "PDF" or "general"
    ↓
    ├─→ PDF Flow:
    │   1. Retrieve (vector search)
    │   2. Generate (RAG answer)
    │   3. Refine (polish response)
    │
    └─→ General Flow:
        1. Generate (general knowledge)
```

### LangSmith Tracing

All nodes are automatically traced:
- Model invocations
- Retrieval operations
- State transitions
- Error handling

## 🔧 Configuration

### Adjust Tracing

To disable tracing temporarily:
```python
# In .env
LANGCHAIN_TRACING_V2=false
```

### Change Project Name

```python
# In .env
LANGCHAIN_PROJECT=my-custom-project
```

### Customize Graph

Modify nodes in the code:
- `classify_question()` - Classification logic
- `retrieve_documents()` - Retrieval parameters
- `generate_answer()` - RAG prompts
- `refine_answer()` - Refinement prompts
- `generate_general_answer()` - General responses

## 📊 Monitoring Best Practices

1. **Tag runs** with metadata (user_id, session_id)
2. **Review failed traces** regularly
3. **Monitor latency** for each node
4. **Track token usage** to optimize costs
5. **Compare model versions** using A/B testing

## 🐛 Troubleshooting

### LangSmith not tracing?

- Verify `LANGSMITH_API_KEY` is correct
- Check `LANGCHAIN_TRACING_V2=true`
- Ensure project exists in LangSmith dashboard

### FastAPI CORS issues?

CORS is enabled for all origins. To restrict:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend
    ...
)
```

### ChromaDB persistence issues?

Ensure `chroma_db` directory has write permissions:
```bash
chmod -R 755 chroma_db
```

## 📝 Notes

- **Thread IDs**: Used for conversation memory
- **Classification**: Determines PDF vs general flow
- **Embeddings**: Cached in `chroma_db/`
- **Memory**: In-memory (InMemorySaver) - resets on restart

## 🚀 Production Deployment

### Environment Variables

- Set proper API keys
- Use production LangSmith project
- Configure CORS properly
- Set up logging


MIT License - feel free to use and modify!
