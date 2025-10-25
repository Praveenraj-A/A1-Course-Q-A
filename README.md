# Course Q&A Chatbot 🤖

An intelligent document-based question-answering system that allows students to upload course materials and get instant, context-aware answers with proper citations.

## 🌟 Features

- **Multi-Format Support**: Upload PDF, DOCX, Excel, CSV, and TXT files
- **AI-Powered Answers**: Uses Google Gemini AI for intelligent responses
- **Semantic Search**: Finds relevant content using meaning, not just keywords
- **Citation System**: Every answer includes references to source documents
- **Real-time Processing**: Fast responses with performance monitoring
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- MongoDB Atlas account
- Google AI Studio API key

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd course-qa-chatbot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create a `.env` file:
```env
MONGO_URI=your_mongodb_atlas_connection_string
GOOGLE_API_KEY=your_google_ai_studio_key
MONGO_DB=courseqa
MONGO_COLLECTION=documents
PORT=8000
```

4. **Run the application**
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

### Endpoints

#### **Upload Documents**
```http
POST /api/v1/upload
```
Upload course materials for processing. Supports PDF, DOCX, XLSX, CSV, and TXT files.

**Response:**
```json
{
  "message": "Successfully processed document.pdf. Created 15 chunks.",
  "document_id": "uuid",
  "processing_type": "PDF extraction",
  "chunks_processed": 15,
  "chunks_failed": 0
}
```

#### **Ask Questions**
```http
GET /api/v1/answer?query=your+question&top_k=5
```
Get AI-generated answers based on uploaded documents.

**Response:**
```json
{
  "answer": "Based on the course materials... [S1]",
  "citations": [
    {
      "source_id": "doc1_chunk3",
      "span": "pg2",
      "confidence": 0.87,
      "section": "Introduction",
      "page_number": 2
    }
  ],
  "confidence": 0.87,
  "document_count": 25,
  "relevant_docs": 3,
  "latency_ms": 1450
}
```

#### **System Monitoring**
```http
GET /api/v1/health          # System health status
GET /api/v1/metrics/kpis    # Performance indicators
GET /api/v1/metrics/system  # System metrics
GET /api/v1/metrics/cost    # Cost analytics
```

## 🏗️ Architecture

### System Overview
```
User → Frontend → FastAPI Backend → AI Services → MongoDB
                     │
                     ├── Document Processing
                     ├── Semantic Search
                     ├── Gemini AI Integration
                     └── Performance Monitoring
```

### Key Components

1. **Document Processing Pipeline**
   - Multi-format text extraction
   - Intelligent chunking with context preservation
   - Vector embedding generation

2. **AI-Powered Search**
   - Semantic search using sentence transformers
   - Hybrid retrieval (vector + text search)
   - Question analysis and type detection

3. **Answer Generation**
   - Google Gemini AI integration
   - Context-aware response generation
   - Automatic citation linking

4. **Monitoring & Analytics**
   - Real-time performance tracking
   - Cost calculation and optimization
   - System health monitoring

## 🔧 Configuration

### Environment Variables
| Variable | Description | Required |
|----------|-------------|----------|
| `MONGO_URI` | MongoDB Atlas connection string | Yes |
| `GOOGLE_API_KEY` | Google AI Studio API key | Yes |
| `MONGO_DB` | Database name | No (default: courseqa) |
| `MONGO_COLLECTION` | Collection name | No (default: documents) |
| `PORT` | Server port | No (default: 8000) |

### File Size Limits
- Maximum file size: 50MB
- Supported formats: PDF, DOCX, XLSX, CSV, TXT
- Recommended chunk size: 800 characters with 100-character overlap

## 📊 Performance Metrics

The system tracks:
- **Response Times**: Average query processing time
- **Success Rates**: Percentage of successful answers
- **Confidence Scores**: AI confidence in generated answers
- **Cost Analytics**: API usage and cost estimation
- **System Health**: Memory, CPU, and uptime monitoring

## 🛠️ Development

### Project Structure
```
course-qa-chatbot/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables
└── README.md             # This file
```

### Key Dependencies
- **FastAPI**: Modern web framework for APIs
- **PyPDF2**: PDF text extraction
- **python-docx**: DOCX document processing
- **pandas**: Excel and CSV file handling
- **sentence-transformers**: Text embeddings
- **google-generativeai**: Gemini AI integration
- **pymongo**: MongoDB database driver

### Running in Development
```bash
# Install in development mode
pip install -e .

# Run with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 🚀 Deployment

### Production Deployment
1. Set up MongoDB Atlas cluster
2. Configure environment variables
3. Deploy using:
   - **Docker** (recommended)
   - **Cloud platforms** (AWS, GCP, Azure)
   - **Traditional servers**

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🔍 Usage Examples

### Example 1: Upload Course Materials
```bash
curl -X POST -F "file=@lecture_notes.pdf" http://localhost:8000/api/v1/upload
```

### Example 2: Ask Questions
```bash
curl "http://localhost:8000/api/v1/answer?query=How%20do%20I%20navigate%20browsers%20with%20Selenium&top_k=5"
```

### Example 3: Check System Health
```bash
curl http://localhost:8000/api/v1/health
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Check the [API Documentation](http://localhost:8000/docs) when running locally
- Open an issue on GitHub
- Contact the development team

## 🙏 Acknowledgments

- Google Gemini AI for advanced language model capabilities
- MongoDB Atlas for scalable database infrastructure
- FastAPI community for excellent web framework documentation
- SentenceTransformers for semantic search capabilities

---

**Note**: This project is designed for educational purposes. Always ensure you have proper rights to upload and process documents.
