import os
import uuid
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import PyPDF2
import docx
import csv
import io
from pymongo import MongoClient
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer, CrossEncoder
import requests
import numpy as np
import re
import hashlib
from bson import ObjectId
from contextlib import asynccontextmanager

# Initialize FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await initialize_services()
    yield
    # Shutdown
    await close_connections()

app = FastAPI(
    title="Course Q&A API",
    description="Course Q&A Chatbot with MongoDB and Pinecone Vector Search",
    version="3.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://praveenrajacse2022_db_user:praveen_199@cluster0.v4wmkh4.mongodb.net/courseqa?retryWrites=true&w=majority&appName=Cluster0")
MONGO_DB = os.getenv("MONGO_DB", "courseqa")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "documents")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_2ndCht_DdDuNQGy7q5NGYLukqW5btSYpAqQcsW9LDd3fF6TzrJzicsUGuFhE8RPKf3X4gg")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "courseqa")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Global variables for services
embedding_model = None
cross_encoder = None
pinecone_index = None
mongo_client = None
documents_collection = None
services_initialized = False

async def initialize_services():
    """Initialize all services"""
    global services_initialized
    
    print("🚀 Starting service initialization...")
    
    # Initialize MongoDB
    await initialize_mongodb()
    
    # Initialize Pinecone
    await initialize_pinecone()
    
    # Initialize ML models
    await initialize_ml_models()
    
    services_initialized = True
    print("✅ All services initialized successfully!")

async def initialize_mongodb(max_retries=3, retry_delay=2):
    """Initialize MongoDB connection with retry logic"""
    global mongo_client, documents_collection
    
    for attempt in range(max_retries):
        try:
            print(f"🔧 Attempting MongoDB connection (attempt {attempt + 1}/{max_retries})...")
            
            mongo_client = MongoClient(
                MONGO_URI, 
                serverSelectionTimeoutMS=10000,
                connectTimeoutMS=10000,
                socketTimeoutMS=10000,
                retryWrites=True,
                retryReads=True
            )
            
            # Test connection
            mongo_client.admin.command('ping')
            print("✅ MongoDB connection test successful")
            
            # Get database and collection
            db = mongo_client[MONGO_DB]
            documents_collection = db[MONGO_COLLECTION]
            
            # Create indexes
            await create_mongodb_indexes()
            
            print("✅ MongoDB initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ MongoDB connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"🔄 Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("❌ All MongoDB connection attempts failed")
                mongo_client = None
                documents_collection = None
                return False

async def create_mongodb_indexes():
    """Create MongoDB indexes with error handling"""
    try:
        if documents_collection is None:
            return
            
        existing_indexes = list(documents_collection.list_indexes())
        existing_index_names = [idx['name'] for idx in existing_indexes]
        
        # Create text index if it doesn't exist
        if 'content_text' not in existing_index_names:
            try:
                documents_collection.create_index([("content", "text")])
                print("✅ Created text index on 'content' field")
            except Exception as e:
                print(f"⚠️  Could not create text index: {e}")
        else:
            print("✅ Text index already exists")
        
        # Create chunk_id index
        if 'chunk_id_1' not in existing_index_names:
            try:
                documents_collection.create_index([("chunk_id", 1)], unique=True)
                print("✅ Created unique index on 'chunk_id' field")
            except Exception as e:
                print(f"⚠️  Could not create chunk_id index: {e}")
        else:
            print("✅ chunk_id index already exists")
            
    except Exception as e:
        print(f"⚠️  Error creating indexes: {e}")

async def initialize_pinecone():
    """Initialize Pinecone connection"""
    global pinecone_index
    
    try:
        if PINECONE_API_KEY and PINECONE_API_KEY != "your-pinecone-api-key-here":
            pc = Pinecone(api_key=PINECONE_API_KEY)
            
            # Check if index exists
            if PINECONE_INDEX_NAME in pc.list_indexes().names():
                pinecone_index = pc.Index(PINECONE_INDEX_NAME)
                print(f"✅ Pinecone index '{PINECONE_INDEX_NAME}' connected")
            else:
                print(f"⚠️  Pinecone index '{PINECONE_INDEX_NAME}' not found")
                pinecone_index = None
        else:
            print("⚠️  Pinecone API key not configured")
            pinecone_index = None
            
    except Exception as e:
        print(f"❌ Pinecone initialization error: {e}")
        pinecone_index = None

async def initialize_ml_models():
    """Initialize ML models"""
    global embedding_model, cross_encoder
    
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded")
    except Exception as e:
        print(f"❌ Embedding model loading error: {e}")
        embedding_model = None
    
    try:
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("✅ Cross-encoder model loaded")
    except Exception as e:
        print(f"❌ Cross-encoder loading error: {e}")
        cross_encoder = None

async def close_connections():
    """Close all connections"""
    if mongo_client:
        mongo_client.close()
        print("✅ MongoDB connection closed")
    print("✅ All connections closed")

def check_database_available():
    """Check if database is available - FIXED VERSION"""
    # FIXED: Don't use truthy check on collection object
    if documents_collection is None:
        return False
    try:
        # Test the connection by pinging the database
        documents_collection.database.command('ping')
        return True
    except Exception as e:
        print(f"Database connection test failed: {e}")
        return False

# Pydantic models
class AnswerRequest(BaseModel):
    query: str
    lang: str = "en"
    top_k: int = 5

class AnswerResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float
    document_count: int
    relevant_docs: int
    latency_ms: float

class UploadResponse(BaseModel):
    message: str
    document_id: str
    processing_type: str
    chunks_processed: int

class DocumentCountResponse(BaseModel):
    document_count: int

class SourceResponse(BaseModel):
    text: str
    metadata: Dict[str, Any]
    source_id: str

# Utility functions
def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file"""
    try:
        doc_file = io.BytesIO(file_content)
        doc = docx.Document(doc_file)
        text = ""
        for paragraph in doc.paragraphs:
            if paragraph.text:
                text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading DOCX: {str(e)}")

def extract_text_from_csv(file_content: bytes) -> str:
    """Extract text from CSV file"""
    try:
        csv_file = io.StringIO(file_content.decode('utf-8'))
        csv_reader = csv.reader(csv_file)
        text = ""
        for row in csv_reader:
            if row:
                text += ", ".join(row) + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

def extract_text_from_txt(file_content: bytes) -> str:
    """Extract text from TXT file"""
    try:
        return file_content.decode('utf-8').strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading text file: {str(e)}")

def chunk_text_intelligently(text: str, filename: str, chunk_size: int = 800, overlap: int = 100) -> List[Dict[str, Any]]:
    """Improved chunking that preserves context and structure"""
    if not text or not text.strip():
        return []
    
    chunks = []
    
    # Clean and normalize text
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # First, try to split by major sections (headings, etc.)
    section_pattern = r'(\n\s*(?:\d+\.\s+.*|\b(?:CHAPTER|SECTION|TOPIC)\s+\d+.*|\b[A-Z][A-Z\s]+\b)\s*\n)'
    sections = re.split(section_pattern, text)
    
    current_section = "Introduction"
    chunk_id = 0
    
    if len(sections) > 1:
        # Process sections with their content
        for i in range(1, len(sections), 2):
            if i < len(sections):
                section_title = sections[i].strip()
                section_content = sections[i+1] if i+1 < len(sections) else ""
                current_section = section_title
                
                # Process this section's content
                section_chunks = chunk_by_semantic_units(section_content, chunk_size, overlap)
                
                for chunk_content in section_chunks:
                    if chunk_content.strip():
                        chunk = create_chunk(chunk_content, filename, current_section, chunk_id)
                        chunks.append(chunk)
                        chunk_id += 1
    else:
        # No clear sections found, chunk the entire text
        content_chunks = chunk_by_semantic_units(text, chunk_size, overlap)
        for chunk_content in content_chunks:
            if chunk_content.strip():
                chunk = create_chunk(chunk_content, filename, "Content", chunk_id)
                chunks.append(chunk)
                chunk_id += 1
    
    return chunks

def chunk_by_semantic_units(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Chunk text by paragraphs, sentences, or fixed size while preserving meaning"""
    chunks = []
    
    # First try to split by paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # If paragraph is very long, split by sentences
        if len(paragraph) > chunk_size:
            sentences = re.split(r'[.!?]+', paragraph)
            current_sentence_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                if len(current_sentence_chunk) + len(sentence) > chunk_size and current_sentence_chunk:
                    chunks.append(current_sentence_chunk.strip())
                    # Start new chunk with overlap
                    sentences_in_chunk = current_sentence_chunk.split('.')
                    if len(sentences_in_chunk) > 1:
                        current_sentence_chunk = '.'.join(sentences_in_chunk[-2:]) + '. ' + sentence
                    else:
                        current_sentence_chunk = sentence
                else:
                    if current_sentence_chunk:
                        current_sentence_chunk += '. ' + sentence
                    else:
                        current_sentence_chunk = sentence
            
            if current_sentence_chunk:
                chunks.append(current_sentence_chunk.strip())
                
        else:
            # Regular paragraph processing
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap from previous
                current_chunk = get_overlap_text(current_chunk, overlap) + "\n\n" + paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def get_overlap_text(text: str, overlap_chars: int) -> str:
    """Get the last overlap_chars characters from text, breaking at sentence boundary if possible"""
    if len(text) <= overlap_chars:
        return text
    
    overlap_start = len(text) - overlap_chars
    # Try to find a sentence boundary in the overlap region
    for i in range(overlap_start, max(0, overlap_start - 100), -1):
        if text[i] in '.!?':
            return text[i+1:].strip()
    
    # If no sentence boundary found, just take the last overlap_chars
    return text[-overlap_chars:].strip()

def create_chunk(content: str, filename: str, section: str, chunk_id: int) -> Dict[str, Any]:
    """Create a standardized chunk object"""
    return {
        "chunk_id": f"{filename}_{chunk_id}",
        "content": content,
        "section": section,
        "filename": filename,
        "page_number": 1,
        "created_at": datetime.utcnow(),
        "metadata": {
            "section": section,
            "filename": filename,
            "chunk_index": chunk_id,
            "content_length": len(content),
            "word_count": len(content.split())
        }
    }

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for text"""
    if embedding_model is None:
        # Fallback: return random embedding
        np.random.seed(hash(text) % 2**32)
        return np.random.rand(384).tolist()
    return embedding_model.encode(text).tolist()

async def openrouter_chat_completion(messages: List[Dict[str, str]], max_tokens: int = 1000) -> Dict[str, Any]:
    """OpenRouter integration with better error handling"""
    if not OPENROUTER_API_KEY:
        return {"error": "OpenRouter API key not configured"}
    
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": "meta-llama/llama-3.1-8b-instruct:free",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,  # Lower temperature for more consistent answers
            "top_p": 0.9,
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=45
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            error_msg = f"OpenRouter API error: {response.status_code}"
            if response.text:
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', {}).get('message', error_msg)
                except:
                    error_msg = f"{error_msg} - {response.text}"
            return {"error": error_msg}
            
    except Exception as e:
        return {"error": f"OpenRouter request failed: {str(e)}"}

def enhanced_text_search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Enhanced text search with multiple strategies"""
    try:
        if documents_collection is None:  # FIXED: Check for None explicitly
            return []
        
        # Clean and preprocess query
        clean_query = re.sub(r'[^\w\s]', ' ', query.lower()).strip()
        query_words = clean_query.split()
        
        # Build multiple search strategies
        search_filters = []
        
        # Strategy 1: Exact phrase match
        if len(query_words) > 1:
            search_filters.append({"content": {"$regex": query, "$options": "i"}})
        
        # Strategy 2: All words (AND)
        if len(query_words) > 1:
            and_filter = {"$and": [{"content": {"$regex": word, "$options": "i"}} for word in query_words]}
            search_filters.append(and_filter)
        
        # Strategy 3: Any word (OR) - for broader matching
        or_filter = {"$or": [{"content": {"$regex": word, "$options": "i"}} for word in query_words]}
        search_filters.append(or_filter)
        
        # Strategy 4: Conceptual matches for common questions
        conceptual_matches = {
            "what is": ["definition", "explain", "meaning", "concept"],
            "how to": ["steps", "procedure", "process", "method"],
            "key topics": ["topics", "subjects", "themes", "areas"],
            "explain": ["explanation", "description", "overview", "introduction"]
        }
        
        for pattern, concepts in conceptual_matches.items():
            if pattern in query.lower():
                for concept in concepts:
                    search_filters.append({"content": {"$regex": concept, "$options": "i"}})
        
        # Combine all strategies
        combined_results = []
        seen_chunks = set()
        
        for search_filter in search_filters:
            try:
                results = list(documents_collection.find(
                    search_filter,
                    {
                        "chunk_id": 1,
                        "content": 1,
                        "section": 1,
                        "filename": 1,
                        "page_number": 1,
                        "metadata": 1
                    }
                ).limit(top_k))
                
                for result in results:
                    chunk_id = result["chunk_id"]
                    if chunk_id not in seen_chunks:
                        seen_chunks.add(chunk_id)
                        # Calculate relevance score
                        content_lower = result["content"].lower()
                        score = calculate_relevance_score(content_lower, query)
                        result["score"] = score
                        combined_results.append(result)
                        
            except Exception as e:
                print(f"Search filter failed: {e}")
                continue
        
        # Sort by score and return top results
        combined_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return combined_results[:top_k]
            
    except Exception as e:
        print(f"Error in enhanced text search: {e}")
        return []

def calculate_relevance_score(content: str, query: str) -> float:
    """Calculate relevance score between content and query"""
    score = 0.0
    query_lower = query.lower()
    content_lower = content.lower()
    
    # Exact phrase match
    if query_lower in content_lower:
        score += 2.0
    
    # Word matches
    query_words = set(re.findall(r'\w+', query_lower))
    content_words = set(re.findall(r'\w+', content_lower))
    
    matching_words = query_words.intersection(content_words)
    if matching_words:
        score += len(matching_words) * 0.3
    
    # Conceptual matches
    conceptual_boosters = {
        "what is": ["definition", "is defined", "means", "refers to"],
        "how to": ["steps", "procedure", "follow these", "method"],
        "explain": ["explanation", "in detail", "overview", "description"]
    }
    
    for pattern, boosters in conceptual_boosters.items():
        if pattern in query_lower:
            for booster in boosters:
                if booster in content_lower:
                    score += 0.5
    
    # Section heading bonus
    if any(marker in content_lower for marker in [":", " - ", "—"]):
        score += 0.3
    
    return min(score, 3.0)  # Cap at 3.0

def hybrid_retrieval(query: str, top_k: int = 8) -> List[Dict[str, Any]]:
    """Enhanced hybrid retrieval with better ranking"""
    
    # Get text search results
    text_results = enhanced_text_search(query, top_k * 2)
    
    # If Pinecone is available, combine with vector search
    if pinecone_index is not None and embedding_model is not None:
        try:
            query_embedding = generate_embedding(query)
            
            vector_results = pinecone_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Process vector results
            vector_chunks = []
            for match in vector_results['matches']:
                chunk_id = match['id']
                if documents_collection is not None:  # FIXED: Check for None explicitly
                    doc = documents_collection.find_one({"chunk_id": chunk_id})
                    if doc:
                        doc['score'] = match['score'] * 1.5  # Boost vector scores
                        vector_chunks.append(doc)
            
            # Combine results
            all_results = []
            seen_chunks = set()
            
            # Add vector results first (usually more relevant)
            for result in vector_chunks:
                chunk_id = result["chunk_id"]
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    all_results.append(result)
            
            # Add text results that weren't already included
            for result in text_results:
                chunk_id = result["chunk_id"]
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    all_results.append(result)
            
            # Sort by score
            all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            return all_results[:top_k]
            
        except Exception as e:
            print(f"Vector search failed: {e}")
    
    # Fallback to text search only
    return text_results[:top_k]

def rerank_results(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Re-rank results using cross-encoder"""
    if not results or cross_encoder is None:
        return results
    
    try:
        pairs = [(query, result["content"]) for result in results]
        scores = cross_encoder.predict(pairs)
        
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])
        
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results
    except Exception as e:
        print(f"Error in reranking: {e}")
        return results

async def generate_comprehensive_answer(query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive answer using available context"""
    
    if not context_chunks:
        # Even without specific context, try to answer using general knowledge
        return await generate_fallback_answer(query)
    
    # Prepare comprehensive context
    context_parts = []
    for i, chunk in enumerate(context_chunks[:6]):  # Use more chunks
        source_id = f"S{i+1}"
        context_parts.append(f"[Source {source_id}] {chunk['content']}")
    
    context = "\n\n".join(context_parts)
    
    # Enhanced prompt for better answers
    prompt = f"""
    You are an expert course teaching assistant. Answer the student's question comprehensively using the provided course materials.

    QUESTION: {query}

    COURSE MATERIALS:
    {context}

    INSTRUCTIONS:
    1. Provide a clear, comprehensive answer to the question
    2. Use the course materials above as your primary source
    3. If the course materials contain relevant information, synthesize it into a coherent answer
    4. If the course materials don't fully answer the question but contain related information, use that as a starting point and provide additional context
    5. Always cite your sources using [S1], [S2], etc. when using specific information from the materials
    6. Structure your answer logically with clear explanations
    7. If providing code examples, ensure they are complete and well-formatted
    8. For conceptual questions, provide definitions and examples

    ANSWER:
    """
    
    # Try OpenRouter API
    openrouter_response = await openrouter_chat_completion([
        {"role": "system", "content": "You are a knowledgeable and helpful course assistant. Provide detailed, accurate answers with proper citations."},
        {"role": "user", "content": prompt}
    ])
    
    if "error" not in openrouter_response and "choices" in openrouter_response:
        answer = openrouter_response["choices"][0]["message"]["content"]
        
        # Extract citations
        citations = extract_citations(answer, context_chunks)
        
        # Calculate confidence based on context relevance
        confidence = calculate_answer_confidence(context_chunks, query)
        
        return {
            "answer": answer,
            "citations": citations,
            "confidence": confidence
        }
    else:
        # Fallback to enhanced simple answer
        return generate_enhanced_simple_answer(query, context_chunks)

async def generate_fallback_answer(query: str) -> Dict[str, Any]:
    """Generate answer when no specific context is found"""
    
    # Try to answer using general knowledge
    prompt = f"""
    You are a helpful course assistant. A student asked: "{query}"
    
    While I couldn't find specific information about this in the uploaded course materials, please provide a helpful and accurate answer based on your general knowledge.
    
    If this is a technical question about programming, testing, or related topics, provide a clear explanation.
    
    Please be honest about the limitations and suggest that the student consult their course materials or instructor for specific details.
    
    Answer:
    """
    
    openrouter_response = await openrouter_chat_completion([
        {"role": "system", "content": "You are a knowledgeable assistant who provides helpful information while being honest about limitations."},
        {"role": "user", "content": prompt}
    ])
    
    if "error" not in openrouter_response and "choices" in openrouter_response:
        answer = openrouter_response["choices"][0]["message"]["content"]
        return {
            "answer": answer,
            "citations": [],
            "confidence": 0.3
        }
    else:
        return {
            "answer": "I don't have specific information about this in the course materials, and I'm currently unable to provide a general knowledge answer. Please try rephrasing your question or consult your course materials directly.",
            "citations": [],
            "confidence": 0.1
        }

def generate_enhanced_simple_answer(query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate enhanced simple answer without LLM"""
    if not context_chunks:
        return {
            "answer": "I couldn't find specific information about this in the course materials. Please try rephrasing your question or ask about a different topic.",
            "citations": [],
            "confidence": 0.1
        }
    
    # Use multiple chunks to create a more comprehensive answer
    answer_parts = []
    
    # Group chunks by section for better organization
    sections = {}
    for chunk in context_chunks[:4]:
        section = chunk.get("section", "General")
        if section not in sections:
            sections[section] = []
        sections[section].append(chunk)
    
    for section_name, section_chunks in sections.items():
        if section_name != "General":
            answer_parts.append(f"**{section_name}**")
        
        for i, chunk in enumerate(section_chunks):
            content = chunk["content"]
            # Clean up the content
            content = re.sub(r'\s+', ' ', content).strip()
            answer_parts.append(f"- {content}")
    
    answer = "Based on the course materials:\n\n" + "\n\n".join(answer_parts)
    
    # Create citations
    citations = []
    for i, chunk in enumerate(context_chunks[:4]):
        citations.append({
            "source_id": chunk["chunk_id"],
            "span": "pg1",
            "confidence": chunk.get("rerank_score", 0.7),
            "section": chunk.get("section", "Unknown"),
            "page_number": 1
        })
    
    return {
        "answer": answer,
        "citations": citations,
        "confidence": 0.6
    }

def extract_citations(answer: str, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract citations from answer text"""
    citations = []
    citation_pattern = r'\[S(\d+)\]'
    matches = re.findall(citation_pattern, answer)
    
    for match in matches:
        source_num = int(match)
        if source_num <= len(context_chunks):
            source_chunk = context_chunks[source_num - 1]
            citations.append({
                "source_id": source_chunk["chunk_id"],
                "span": "pg1",
                "confidence": source_chunk.get("rerank_score", 0.8),
                "section": source_chunk.get("section", "Unknown"),
                "page_number": 1
            })
    
    # If no citations found but we have context, add the most relevant one
    if not citations and context_chunks:
        citations.append({
            "source_id": context_chunks[0]["chunk_id"],
            "span": "pg1",
            "confidence": context_chunks[0].get("rerank_score", 0.8),
            "section": context_chunks[0].get("section", "Unknown"),
            "page_number": 1
        })
    
    return citations

def calculate_answer_confidence(context_chunks: List[Dict[str, Any]], query: str) -> float:
    """Calculate confidence score for the answer"""
    if not context_chunks:
        return 0.1
    
    # Base confidence from retrieval scores
    max_score = max([chunk.get("rerank_score", 0.5) for chunk in context_chunks])
    base_confidence = min(max_score, 0.9)
    
    # Boost for conceptual questions with good context
    conceptual_indicators = ["what is", "explain", "define", "how does", "what are"]
    if any(indicator in query.lower() for indicator in conceptual_indicators):
        # Check if we have definition-like content
        definition_indicators = ["is defined", "means", "refers to", "definition", "concept"]
        for chunk in context_chunks:
            content_lower = chunk["content"].lower()
            if any(indicator in content_lower for indicator in definition_indicators):
                base_confidence = min(base_confidence + 0.1, 0.95)
                break
    
    return round(base_confidence, 2)

# API Routes
@app.get("/")
async def root():
    return {
        "message": "Course Q&A Chatbot API with Enhanced Search", 
        "status": "running",
        "version": "3.0.0",
        "features": [
            "Intelligent Text Chunking",
            "Enhanced Semantic Search", 
            "Comprehensive Answer Generation",
            "Multi-strategy Retrieval",
            "Context-Aware Responses"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "mongo": check_database_available(),
        "pinecone": pinecone_index is not None,
        "embedding_model": embedding_model is not None,
        "cross_encoder": cross_encoder is not None,
        "openrouter": bool(OPENROUTER_API_KEY),
        "services_initialized": services_initialized
    }
    
    overall_status = "healthy" if status["mongo"] else "degraded"
    if not status["mongo"]:
        overall_status = "critical"
    
    return {
        "status": overall_status, 
        "services": status,
        "version": "3.0.0"
    }

@app.post("/api/v1/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload and process course material file"""
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not check_database_available():
        raise HTTPException(status_code=503, detail="Database service temporarily unavailable")
    
    file_content = await file.read()
    
    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="Empty file provided")
    
    filename = file.filename.lower()
    
    try:
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_content)
            processing_type = "PDF extraction"
        elif filename.endswith('.docx'):
            text = extract_text_from_docx(file_content)
            processing_type = "DOCX extraction"
        elif filename.endswith('.csv'):
            text = extract_text_from_csv(file_content)
            processing_type = "CSV extraction"
        elif filename.endswith('.txt') or filename.endswith('.md'):
            text = extract_text_from_txt(file_content)
            processing_type = "Text extraction"
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="No text content could be extracted")
        
        print(f"📄 Extracted {len(text)} characters from {file.filename}")
        
        # Use intelligent chunking
        chunks = chunk_text_intelligently(text, file.filename)
        print(f"📦 Created {len(chunks)} intelligent chunks from {file.filename}")
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No meaningful content chunks could be created")
        
        # Process chunks
        processed_count = 0
        for chunk in chunks:
            try:
                embedding = generate_embedding(chunk["content"])
                
                # Store in MongoDB
                mongo_doc = chunk.copy()
                documents_collection.insert_one(mongo_doc)
                
                # Store in Pinecone if available
                if pinecone_index is not None:
                    try:
                        pinecone_index.upsert(
                            vectors=[{
                                "id": chunk["chunk_id"],
                                "values": embedding,
                                "metadata": {
                                    "filename": chunk["filename"],
                                    "section": chunk["section"],
                                    "content_preview": chunk["content"][:150]
                                }
                            }]
                        )
                    except Exception as e:
                        print(f"⚠️  Failed to store in Pinecone: {e}")
                
                processed_count += 1
            except Exception as e:
                print(f"Error processing chunk {chunk['chunk_id']}: {e}")
                continue
        
        return UploadResponse(
            message=f"Successfully processed {file.filename}. Created {processed_count} intelligent chunks.",
            document_id=str(uuid.uuid4()),
            processing_type=processing_type,
            chunks_processed=processed_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in file upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/api/v1/documents/count", response_model=DocumentCountResponse)
async def get_document_count():
    """Get count of processed documents"""
    try:
        if not check_database_available():
            return DocumentCountResponse(document_count=0)
        count = documents_collection.count_documents({})
        return DocumentCountResponse(document_count=count)
    except Exception as e:
        return DocumentCountResponse(document_count=0)

@app.delete("/api/v1/documents")
async def clear_all_documents():
    """Clear all documents from both MongoDB and Pinecone"""
    try:
        if not check_database_available():
            return {"message": "Database service not available"}
        
        chunks = list(documents_collection.find({}, {"chunk_id": 1}))
        chunk_ids = [chunk["chunk_id"] for chunk in chunks] if chunks else []
        
        # Delete from MongoDB
        mongo_result = documents_collection.delete_many({})
        
        # Delete from Pinecone if available
        pinecone_deleted = 0
        if chunk_ids and pinecone_index is not None:
            try:
                pinecone_index.delete(ids=chunk_ids)
                pinecone_deleted = len(chunk_ids)
            except Exception as e:
                print(f"Error deleting from Pinecone: {e}")
        
        return {
            "message": f"Successfully cleared {mongo_result.deleted_count} documents from MongoDB and {pinecone_deleted} vectors from Pinecone"
        }
    except Exception as e:
        print(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing documents: {str(e)}")

@app.get("/api/v1/answer", response_model=AnswerResponse)
async def get_answer(
    query: str,
    lang: str = "en",
    top_k: int = 8  # Increased for better coverage
):
    """Get comprehensive answer to course-related question"""
    start_time = time.time()
    
    try:
        if not check_database_available():
            raise HTTPException(status_code=503, detail="Database service temporarily unavailable")
        
        doc_count = documents_collection.count_documents({})
        
        if doc_count == 0:
            raise HTTPException(status_code=404, detail="No course materials found. Please upload documents first.")
        
        print(f"🔍 Enhanced search for: '{query}'")
        
        # Perform enhanced retrieval
        retrieved_chunks = hybrid_retrieval(query, top_k)
        print(f"📚 Retrieved {len(retrieved_chunks)} relevant chunks")
        
        # Re-rank results
        reranked_chunks = rerank_results(query, retrieved_chunks)
        print(f"🎯 After reranking: {len(reranked_chunks)} high-quality chunks")
        
        # Generate comprehensive answer
        answer_data = await generate_comprehensive_answer(query, reranked_chunks)
        
        latency_ms = (time.time() - start_time) * 1000
        
        print(f"✅ Comprehensive answer generated in {latency_ms:.2f}ms with confidence {answer_data['confidence']}")
        
        return AnswerResponse(
            answer=answer_data["answer"],
            citations=answer_data["citations"],
            confidence=answer_data["confidence"],
            document_count=doc_count,
            relevant_docs=len(reranked_chunks),
            latency_ms=latency_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating answer: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.get("/api/v1/source/{source_id}", response_model=SourceResponse)
async def get_source(source_id: str):
    """Get source document by source_id"""
    try:
        if not check_database_available():
            raise HTTPException(status_code=503, detail="Database service temporarily unavailable")
            
        document = documents_collection.find_one({"chunk_id": source_id})
        if not document:
            raise HTTPException(status_code=404, detail="Source not found")
        
        return SourceResponse(
            text=document["content"],
            metadata=document.get("metadata", {}),
            source_id=source_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving source: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    print(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)