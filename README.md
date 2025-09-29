Course Q&A Chatbot
An intelligent question-answering system that processes course materials and provides accurate, context-aware answers using advanced AI and vector search technologies.
🚀 Features
•	Multi-format Document Support: Upload and process PDF, DOCX, CSV, and TXT files
•	Intelligent Text Processing: Smart chunking that preserves document structure and context
•	Advanced Search: Hybrid retrieval combining semantic search and keyword matching
•	AI-Powered Answers: Google Gemini integration for comprehensive, contextual responses
•	Spelling Variation Tolerance: Automatically handles typos and word variations
•	Citation System: Source tracking with confidence scores and references
•	RESTful API: FastAPI-based with comprehensive endpoints and CORS support
📦 Installation
Prerequisites
•	Python 3.8+
•	MongoDB
•	Pinecone account
•	Google AI Studio API key
Quick Start
1.	Clone and setup
bash
git clone <repository-url>
cd course-qa-chatbot
2.	Install dependencies
bash
pip install -r requirements.txt
3.	Environment Configuration
Create a .env file:
env
MONGO_URI=mongodb://localhost:27017
MONGO_DB=courseqa
PINECONE_API_KEY=your_pinecone_key
GOOGLE_API_KEY=your_gemini_key
PORT=8000
4.	Start the application
bash
python main.py
🚀 Usage
Upload Documents
bash
curl -X POST -F "file=@course_materials.pdf" http://localhost:8000/api/v1/upload
Ask Questions
bash
curl "http://localhost:8000/api/v1/answer?query=explain%20machine%20learning"
API Endpoints
Method	Endpoint	Description
POST	/api/v1/upload	Upload documents
GET	/api/v1/answer	Get answers
GET	/api/v1/documents/count	Document count
DELETE	/api/v1/documents	Clear documents
GET	/health	Health check
🔧 Configuration
Environment Variables
Variable	Description	Default
MONGO_URI	MongoDB connection	mongodb://localhost:27017
PINECONE_API_KEY	Pinecone API key	Required
GOOGLE_API_KEY	Gemini AI API key	Required
PORT	Server port	8000
Model Configuration
•	Embedding Model: all-MiniLM-L6-v2
•	Reranking Model: cross-encoder/ms-marco-MiniLM-L-6-v2
•	LLM: Google Gemini 2.0 Flash
🛠️ Development
Project Structure
text
backend/
├── main.py              # FastAPI application
├── core/               # Configuration & database
├── models/             # Data schemas
├── api/               # Route handlers
├── services/          # Business logic
└── utils/             # Utilities & helpers
Key Components
Intelligent Chunking
•	Preserves document sections and headings
•	Maintains sentence boundaries
•	Configurable chunk sizes with overlap
Hybrid Retrieval
•	Semantic search with vector similarity
•	Keyword search with regex matching
•	Automatic spelling correction
•	Cross-encoder re-ranking
Spelling Tolerance
Handles variations like:
•	Plurals (internship → internships)
•	Common typos (navigations → navigation)
•	Verb forms (developing → development)
📊 Performance
•	File Size: 50MB maximum per upload
•	Response Time: 1-4 seconds typical
•	Document Formats: PDF, DOCX, CSV, TXT, MD
🔍 Troubleshooting
Common Issues
No results for valid queries
•	Verify documents are uploaded
•	Check MongoDB connection
•	Ensure Pinecone index exists
Spelling variations not working
•	Check spellchecker installation
•	Review search strategy logs
Slow response times
•	Optimize chunk sizes
•	Check model loading times
Logs and Monitoring
Enable debug logging:
python
logging.basicConfig(level=logging.DEBUG)
🚀 Deployment
Production Setup
bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
📄 License
MIT License - see LICENSE file for details.
🙏 Acknowledgments
•	Google Gemini AI for language understanding
•	Sentence Transformers for embeddings
•	Pinecone for vector search
•	FastAPI for high-performance API

