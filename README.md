# Course Q&A Chatbot

An AI-powered chatbot that answers course-related questions by processing uploaded documents and using vector search with Google Gemini.

---

## 🚀 Features
- Upload PDF, DOCX, CSV, TXT files
- Smart text chunking and context-aware search
- AI-powered answers with Google Gemini
- REST API built with FastAPI
- MongoDB + Pinecone for storage and vector search

---

## 📦 Installation

### Requirements
- Python 3.8+
- MongoDB
- Pinecone API Key
- Google AI Studio API Key

### Setup
```bash
git clone <repository-url>
cd course-qa-chatbot
pip install -r requirements.txt
Create a .env file:

env
Copy code
MONGO_URI=mongodb://localhost:27017
MONGO_DB=courseqa
PINECONE_API_KEY=your_pinecone_key
GOOGLE_API_KEY=your_gemini_key
PORT=8000
Run the app:

bash
Copy code
python main.py
📖 Usage
Upload a document:

bash
Copy code
curl -X POST -F "file=@notes.pdf" http://localhost:8000/api/v1/upload
Ask a question:

bash
Copy code
curl "http://localhost:8000/api/v1/answer?query=what%20is%20machine%20learning"
🔧 API Endpoints
POST /api/v1/upload → Upload documents

GET /api/v1/answer → Get answers

GET /api/v1/documents/count → Count uploaded docs

DELETE /api/v1/documents → Clear documents

GET /health → Health check

