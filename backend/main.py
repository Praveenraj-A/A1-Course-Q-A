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
import google.generativeai as genai
import numpy as np
import re
import hashlib
from bson import ObjectId
from contextlib import asynccontextmanager
import logging
import asyncio
from spellchecker import SpellChecker

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize spell checker
spell = SpellChecker()

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
    version="4.7.0",
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

# Load environment variables from .env file
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "courseqa")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "documents")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "courseqa")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PORT = int(os.getenv("PORT", "8000"))

# Global variables for services
embedding_model = None
cross_encoder = None
pinecone_index = None
mongo_client = None
documents_collection = None
services_initialized = False
gemini_model = None

async def initialize_services():
    """Initialize all services"""
    global services_initialized, gemini_model
    
    logger.info("🚀 Starting service initialization...")
    
    # Initialize MongoDB
    await initialize_mongodb()
    
    # Initialize Pinecone
    await initialize_pinecone()
    
    # Initialize ML models
    await initialize_ml_models()
    
    # Initialize Gemini model
    await initialize_gemini()
    
    services_initialized = True
    logger.info("✅ All services initialized successfully!")

async def initialize_mongodb(max_retries=3, retry_delay=2):
    """Initialize MongoDB connection with retry logic"""
    global mongo_client, documents_collection
    
    if not MONGO_URI:
        logger.error("❌ MongoDB URI not configured")
        mongo_client = None
        documents_collection = None
        return False
        
    for attempt in range(max_retries):
        try:
            logger.info(f"🔧 Attempting MongoDB connection (attempt {attempt + 1}/{max_retries})...")
            
            mongo_client = MongoClient(
                MONGO_URI, 
                serverSelectionTimeoutMS=15000,
                connectTimeoutMS=15000,
                socketTimeoutMS=15000,
                retryWrites=True,
                retryReads=True
            )
            
            # Test connection
            mongo_client.admin.command('ping')
            logger.info("✅ MongoDB connection test successful")
            
            # Get database and collection
            db = mongo_client[MONGO_DB]
            documents_collection = db[MONGO_COLLECTION]
            
            # Create indexes
            await create_mongodb_indexes()
            
            logger.info("✅ MongoDB initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"🔄 Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("❌ All MongoDB connection attempts failed")
                mongo_client = None
                documents_collection = None
                return False

async def create_mongodb_indexes():
    """Create MongoDB indexes with error handling"""
    try:
        if documents_collection is None:
            return
            
        # Create text index
        try:
            documents_collection.create_index([("content", "text")])
            logger.info("✅ Created text index on 'content' field")
        except Exception as e:
            logger.info(f"✅ Text index already exists or couldn't be created: {e}")
        
        # Create chunk_id index
        try:
            documents_collection.create_index([("chunk_id", 1)], unique=True)
            logger.info("✅ Created unique index on 'chunk_id' field")
        except Exception as e:
            logger.info(f"✅ chunk_id index already exists or couldn't be created: {e}")
            
    except Exception as e:
        logger.error(f"⚠️  Error creating indexes: {e}")

async def initialize_pinecone():
    """Initialize Pinecone connection with better error handling"""
    global pinecone_index
    
    if not PINECONE_API_KEY:
        logger.warning("⚠️  Pinecone API key not configured")
        pinecone_index = None
        return
        
    try:
        logger.info("🔧 Initializing Pinecone...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists
        try:
            existing_indexes = pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes.indexes] if hasattr(existing_indexes, 'indexes') else existing_indexes.names()
            
            if PINECONE_INDEX_NAME in index_names:
                pinecone_index = pc.Index(PINECONE_INDEX_NAME)
                logger.info(f"✅ Pinecone index '{PINECONE_INDEX_NAME}' connected")
                
                # Test the connection
                try:
                    pinecone_index.describe_index_stats()
                    logger.info("✅ Pinecone index connection test successful")
                except Exception as e:
                    logger.warning(f"⚠️  Pinecone index connection test failed: {e}")
                    pinecone_index = None
            else:
                logger.warning(f"⚠️  Pinecone index '{PINECONE_INDEX_NAME}' not found")
                logger.info("💡 You can create the index manually in the Pinecone console")
                pinecone_index = None
                
        except Exception as e:
            logger.error(f"❌ Error listing Pinecone indexes: {e}")
            pinecone_index = None
            
    except Exception as e:
        logger.error(f"❌ Pinecone initialization error: {e}")
        pinecone_index = None

async def initialize_ml_models():
    """Initialize ML models"""
    global embedding_model, cross_encoder
    
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Embedding model loaded")
    except Exception as e:
        logger.error(f"❌ Embedding model loading error: {e}")
        embedding_model = None
    
    try:
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        logger.info("✅ Cross-encoder model loaded")
    except Exception as e:
        logger.error(f"❌ Cross-encoder loading error: {e}")
        cross_encoder = None

async def initialize_gemini():
    """Initialize Google Gemini AI model"""
    global gemini_model
    
    if not GOOGLE_API_KEY:
        logger.warning("⚠️  Google AI Studio API key not configured")
        gemini_model = None
        return
        
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        logger.info("✅ Google Gemini model initialized")
        
    except Exception as e:
        logger.error(f"❌ Gemini initialization error: {e}")
        gemini_model = None

async def close_connections():
    """Close all connections"""
    if mongo_client:
        mongo_client.close()
        logger.info("✅ MongoDB connection closed")
    logger.info("✅ All connections closed")

def check_database_available():
    """Check if database is available"""
    if documents_collection is None:
        return False
    try:
        documents_collection.database.command('ping')
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
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
    chunks_failed: int

class DocumentCountResponse(BaseModel):
    document_count: int

class SourceResponse(BaseModel):
    text: str
    metadata: Dict[str, Any]
    source_id: str

# Enhanced text processing functions
def normalize_text(text: str) -> str:
    """Normalize text for better matching - handle plurals, verb forms, etc."""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Handle common pluralizations and variations
    variations = {
        r'navigations?\b': 'navigation',
        r'browsers?\b': 'browser',
        r'drivers?\b': 'driver',
        r'automations?\b': 'automation',
        r'clicks?\b': 'click',
        r'opens?\b': 'open',
        r'tabs?\b': 'tab',
        r'windows?\b': 'window',
        r'buttons?\b': 'button',
        r'forms?\b': 'form',
        r'elements?\b': 'element',
        r'pages?\b': 'page',
        r'websites?\b': 'website',
        r'applications?\b': 'application',
        r'methods?\b': 'method',
        r'functions?\b': 'function',
        r'variables?\b': 'variable',
        r'classes?\b': 'class',
        r'objects?\b': 'object',
    }
    
    for pattern, replacement in variations.items():
        text = re.sub(pattern, replacement, text)
    
    return text

def expand_query_terms(query: str) -> list[str]:
    """Expand query with synonyms and related terms"""
    base_terms = query.lower().split()
    expanded_terms = set(base_terms)
    
    # Common synonyms and related terms
    synonyms = {
        "navigate": ["navigation", "browse", "go to", "visit", "open"],
        "browser": ["web browser", "chrome", "firefox", "safari", "edge"],
        "driver": ["webdriver", "selenium driver", "browser driver"],
        "open": ["launch", "start", "load", "initialize"],
        "tab": ["browser tab", "new tab", "window tab"],
        "click": ["press", "select", "tap", "choose"],
        "find": ["locate", "search", "identify", "detect"],
        "get": ["retrieve", "obtain", "fetch", "acquire"],
        "how": ["method", "way", "process", "procedure"],
        "what": ["explain", "describe", "define", "meaning"],
        "selenium": ["web automation", "browser automation", "web testing"],
        "webdriver": ["browser driver", "selenium driver", "automation driver"],
    }
    
    for term in base_terms:
        if term in synonyms:
            expanded_terms.update(synonyms[term])
    
    return list(expanded_terms)

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts using word overlap and pattern matching"""
    if not text1 or not text2:
        return 0.0
    
    # Normalize both texts
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    
    # Split into words
    words1 = set(re.findall(r'\w+', norm1))
    words2 = set(re.findall(r'\w+', norm2))
    
    if not words1 or not words2:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    if union == 0:
        return 0.0
    
    jaccard_similarity = intersection / union
    
    # Bonus for exact phrase matches
    exact_match_boost = 0.0
    if text1.lower() in text2.lower() or text2.lower() in text1.lower():
        exact_match_boost = 0.3
    
    # Bonus for shared important terms
    important_terms = ["navigation", "browser", "driver", "selenium", "webdriver", "click", "open", "tab", "window", "automation"]
    shared_important = len([term for term in important_terms if term in norm1 and term in norm2])
    important_boost = shared_important * 0.1
    
    return min(jaccard_similarity + exact_match_boost + important_boost, 1.0)

# Enhanced Question Analysis with improved typo handling
def analyze_question_type(query: str) -> Dict[str, Any]:
    """Analyze the question to determine its type and requirements with enhanced typo handling"""
    query_lower = query.lower().strip()
    
    # Enhanced typo handling for common misspellings
    typo_corrections = {
        "id": "is", "hwodo": "how do", "whos": "who is", "wats": "what is",
        "navigations": "navigation", "selenimum": "selenium", "webdriverr": "webdriver",
        "brower": "browser", "drivr": "driver", "automat": "automate", "expln": "explain",
        "hw": "how", "wat": "what", "wen": "when", "wer": "where", "y": "why"
    }
    
    for typo, correction in typo_corrections.items():
        query_lower = query_lower.replace(typo, correction)
    
    # Auto-correct spelling using spell checker
    words = query_lower.split()
    corrected_words = []
    for word in words:
        if len(word) > 2:  # Only correct words longer than 2 characters
            correction = spell.correction(word)
            if correction and correction != word:
                corrected_words.append(correction)
                logger.info(f"🔧 Corrected '{word}' to '{correction}'")
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)
    
    query_lower = " ".join(corrected_words)
    
    question_types = {
        "who": any(keyword in query_lower for keyword in ["who is", "who are", "who was", "who's", "this person"]),
        "what": any(keyword in query_lower for keyword in ["what is", "what are", "what does", "what's", "explain"]),
        "how": any(keyword in query_lower for keyword in ["how to", "how do", "how does", "how can", "how i"]),
        "why": any(keyword in query_lower for keyword in ["why", "reason", "purpose"]),
        "contact": any(keyword in query_lower for keyword in ["contact", "email", "phone", "mobile", "number", "reach", "get in touch"]),
        "skills": any(keyword in query_lower for keyword in ["skills", "technologies", "languages", "frameworks", "proficient in"]),
        "projects": any(keyword in query_lower for keyword in ["projects", "project", "work on", "built", "developed", "explain project"]),
        "education": any(keyword in query_lower for keyword in ["education", "degree", "college", "university", "study"]),
        "technical": any(keyword in query_lower for keyword in ["code", "program", "script", "function", "method", "class", "variable"]),
        "navigation": any(keyword in query_lower for keyword in ["navigate", "navigation", "browser", "url", "website", "webpage"]),
    }
    
    # Remove name detection - search names in documents instead
    name_detected = False
    
    # Determine primary question type
    primary_type = next((q_type for q_type, matches in question_types.items() if matches), "general")
    
    # Override for specific queries
    if "mobile number" in query_lower or "phone number" in query_lower:
        primary_type = "contact"
    
    return {
        "type": primary_type,
        "requires_person_info": question_types["who"] or name_detected,
        "requires_contact": question_types["contact"],
        "requires_skills": question_types["skills"],
        "requires_projects": question_types["projects"],
        "requires_education": question_types["education"],
        "requires_technical": question_types["technical"],
        "requires_navigation": question_types["navigation"],
        "name_detected": name_detected,
        "original_query": query,
        "processed_query": query_lower
    }

# Content Extraction Functions
def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file with improved parsing"""
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                # Clean up the text
                page_text = re.sub(r'\s+', ' ', page_text).strip()
                text += f"Page {page_num + 1}: {page_text}\n\n"
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No readable text found in PDF")
            
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
            if paragraph.text and paragraph.text.strip():
                text += paragraph.text.strip() + "\n"
        
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading DOCX: {str(e)}")

def extract_text_from_csv(file_content: bytes) -> str:
    """Extract text from CSV file with better formatting"""
    try:
        csv_file = io.StringIO(file_content.decode('utf-8'))
        csv_reader = csv.reader(csv_file)
        text = ""
        
        for i, row in enumerate(csv_reader):
            if row and any(cell.strip() for cell in row):
                clean_row = [cell.strip() for cell in row if cell.strip()]
                if clean_row:
                    text += f"Row {i+1}: {', '.join(clean_row)}\n"
        
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

def extract_text_from_txt(file_content: bytes) -> str:
    """Extract text from TXT file with encoding detection"""
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                text = file_content.decode(encoding)
                # Clean up the text
                text = re.sub(r'\r\n', '\n', text)
                text = re.sub(r'\n\s*\n', '\n\n', text)
                return text.strip()
            except UnicodeDecodeError:
                continue
        
        raise UnicodeDecodeError("Could not decode file with any supported encoding")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading text file: {str(e)}")

def intelligent_chunking(text: str, filename: str, chunk_size: int = 1200, overlap: int = 150) -> List[Dict[str, Any]]:
    """Improved chunking that preserves complete sentences and paragraphs"""
    if not text or not text.strip():
        return []
    
    chunks = []
    
    # Normalize text
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Split by major sections first (preserve document structure)
    section_pattern = r'(\n#{1,6}\s+.+|\n\s*(?:CHAPTER|SECTION|TOPIC|UNIT)\s+[\dIVX]+.*?|\n\s*\d+\.\s*[A-Z][^\n]+|\n\s*[A-Z][A-Z\s]{10,}[A-Z]\s*\n)'
    sections = re.split(section_pattern, text, flags=re.IGNORECASE)
    
    current_section = "Document Content"
    chunk_id = 0
    
    if len(sections) > 1:
        # Process with sections
        for i in range(0, len(sections), 2):
            if i < len(sections):
                if i + 1 < len(sections):
                    section_title = sections[i].strip() if sections[i].strip() else current_section
                    section_content = sections[i + 1]
                    
                    if section_title and len(section_title) > 2:
                        current_section = section_title
                    
                    if section_content.strip():
                        content_chunks = split_into_meaningful_chunks(section_content, chunk_size, overlap)
                        for chunk_content in content_chunks:
                            if chunk_content.strip():
                                chunk = create_chunk(chunk_content, filename, current_section, chunk_id)
                                chunks.append(chunk)
                                chunk_id += 1
    else:
        # No sections found, split the entire text
        content_chunks = split_into_meaningful_chunks(text, chunk_size, overlap)
        for chunk_content in content_chunks:
            if chunk_content.strip():
                chunk = create_chunk(chunk_content, filename, "Content", chunk_id)
                chunks.append(chunk)
                chunk_id += 1
    
    logger.info(f"Created {len(chunks)} chunks with average length {sum(len(c['content']) for c in chunks) // len(chunks) if chunks else 0} characters")
    return chunks

def split_into_meaningful_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into chunks while preserving sentence and paragraph boundaries"""
    chunks = []
    
    # First, split by paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If current chunk + paragraph would exceed size and current chunk is not empty
        if len(current_chunk) + len(paragraph) + 2 > chunk_size and current_chunk:
            # Add the current chunk to chunks
            chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap from previous chunk
            if overlap > 0:
                # Use last few sentences as overlap
                sentences = re.split(r'(?<=[.!?])\s+', current_chunk)
                overlap_text = ' '.join(sentences[-min(3, len(sentences)):])
                current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
            else:
                current_chunk = paragraph
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add the final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # If chunks are still too large, split by sentences within paragraphs
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > chunk_size * 1.5:
            # Split this chunk by sentences
            sentences = re.split(r'(?<=[.!?])\s+', chunk)
            temp_chunk = ""
            for sentence in sentences:
                if len(temp_chunk) + len(sentence) + 1 > chunk_size and temp_chunk:
                    final_chunks.append(temp_chunk.strip())
                    temp_chunk = sentence
                else:
                    if temp_chunk:
                        temp_chunk += " " + sentence
                    else:
                        temp_chunk = sentence
            if temp_chunk.strip():
                final_chunks.append(temp_chunk.strip())
        else:
            final_chunks.append(chunk)
    
    return final_chunks

def create_chunk(content: str, filename: str, section: str, chunk_id: int) -> Dict[str, Any]:
    """Create a standardized chunk object"""
    # Generate a unique chunk ID
    chunk_hash = hashlib.md5(f"{filename}_{section}_{chunk_id}_{content[:50]}".encode()).hexdigest()[:12]
    chunk_identifier = f"{filename}_{chunk_hash}"
    
    return {
        "chunk_id": chunk_identifier,
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
            "word_count": len(content.split()),
            "chunk_hash": chunk_hash
        }
    }

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for text"""
    if embedding_model is None:
        # Fallback: return zero vector (better than random)
        return [0.0] * 384
    
    try:
        # Limit text length for embedding to avoid token limits
        if len(text) > 2000:
            text = text[:2000]
        return embedding_model.encode(text).tolist()
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return [0.0] * 384

async def gemini_chat_completion(prompt: str, max_tokens: int = 2000) -> Dict[str, Any]:
    """Google Gemini API integration with better error handling"""
    
    if not GOOGLE_API_KEY:
        return {"error": "Google AI Studio API key not configured"}
    
    if gemini_model is None:
        return {"error": "Gemini model not initialized"}
    
    try:
        # Generate content using Gemini API
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.1,  # Lower temperature for more factual answers
                    top_p=0.8,
                )
            )
        )
        
        if response.text:
            return {
                "choices": [
                    {
                        "message": {
                            "content": response.text
                        }
                    }
                ]
            }
        else:
            logger.warning("Empty response from Gemini API")
            return {"error": "Empty response from Gemini API"}
            
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return {"error": f"Google Gemini API request failed: {str(e)}"}

def semantic_search(query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """Enhanced semantic search with better relevance scoring and spelling tolerance"""
    try:
        if documents_collection is None:
            return []
        
        # Normalize and expand query
        normalized_query = normalize_text(query)
        expanded_terms = expand_query_terms(query)
        
        logger.info(f"🔍 Search: '{query}' -> Normalized: '{normalized_query}'")
        logger.info(f"📝 Expanded terms: {expanded_terms}")
        
        # Build search queries with multiple strategies
        search_queries = []
        
        # 1. Exact phrase match (boosted)
        if len(normalized_query.split()) > 1:
            exact_phrase = normalized_query
            search_queries.append({
                "filter": {"content": {"$regex": re.escape(exact_phrase), "$options": "i"}},
                "boost": 15.0,
                "type": "exact_phrase"
            })
        
        # 2. All expanded terms (AND)
        if expanded_terms:
            and_conditions = [{"content": {"$regex": re.escape(term), "$options": "i"}} for term in expanded_terms[:5]]
            if and_conditions:
                search_queries.append({
                    "filter": {"$and": and_conditions},
                    "boost": 8.0,
                    "type": "all_terms"
                })
        
        # 3. Any expanded terms (OR)
        if expanded_terms:
            or_conditions = [{"content": {"$regex": re.escape(term), "$options": "i"}} for term in expanded_terms[:8]]
            if or_conditions:
                search_queries.append({
                    "filter": {"$or": or_conditions},
                    "boost": 3.0,
                    "type": "any_terms"
                })
        
        # 4. Individual term matches with semantic similarity
        individual_terms = normalized_query.split()
        for term in individual_terms[:4]:
            if len(term) > 2:  # Only search for terms longer than 2 characters
                search_queries.append({
                    "filter": {"content": {"$regex": re.escape(term), "$options": "i"}},
                    "boost": 2.0,
                    "type": f"term_{term}"
                })
        
        # Execute all search strategies
        all_results = []
        seen_chunks = set()
        
        for search_query in search_queries:
            try:
                results = list(documents_collection.find(
                    search_query["filter"],
                    {
                        "chunk_id": 1,
                        "content": 1,
                        "section": 1,
                        "filename": 1,
                        "page_number": 1,
                        "metadata": 1
                    }
                ).limit(top_k * 2))
                
                for result in results:
                    chunk_id = result["chunk_id"]
                    if chunk_id not in seen_chunks:
                        seen_chunks.add(chunk_id)
                        # Calculate enhanced relevance score with semantic similarity
                        score = calculate_enhanced_relevance(result["content"], query, search_query["boost"])
                        result["score"] = score
                        result["search_type"] = search_query["type"]
                        all_results.append(result)
                        
            except Exception as e:
                logger.warning(f"Search query {search_query['type']} failed: {e}")
                continue
        
        # If no results found, try a more lenient search
        if not all_results:
            logger.info("🔄 No results found, trying lenient search...")
            lenient_results = list(documents_collection.find(
                {"$text": {"$search": query}},
                {
                    "chunk_id": 1,
                    "content": 1,
                    "section": 1,
                    "filename": 1,
                    "page_number": 1,
                    "metadata": 1,
                    "score": {"$meta": "textScore"}
                }
            ).sort([("score", {"$meta": "textScore"})]).limit(top_k))
            
            for result in lenient_results:
                chunk_id = result["chunk_id"]
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    result["score"] = result.get("score", 0.5) * 0.7  # Reduce score for text search
                    result["search_type"] = "text_search"
                    all_results.append(result)
        
        # Sort by score and return top results
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        logger.info(f"📚 Found {len(all_results)} total results, returning top {top_k}")
        for i, result in enumerate(all_results[:3]):
            logger.info(f"   {i+1}. Score: {result.get('score', 0):.2f}, Type: {result.get('search_type', 'unknown')}")
        
        return all_results[:top_k]
            
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        return []

def calculate_enhanced_relevance(content: str, query: str, base_boost: float = 1.0) -> float:
    """Calculate enhanced relevance between content and query with semantic similarity"""
    score = 0.0
    query_lower = query.lower()
    content_lower = content.lower()
    
    # Calculate semantic similarity
    semantic_similarity = calculate_semantic_similarity(query, content)
    score += semantic_similarity * 10.0 * base_boost
    
    # Exact phrase match (highest priority)
    if query_lower in content_lower:
        score += 20.0 * base_boost
    
    # Word overlap with term frequency consideration
    query_words = set(re.findall(r'\w+', query_lower))
    content_words = re.findall(r'\w+', content_lower)
    content_word_set = set(content_words)
    
    # Calculate word overlap
    matching_words = query_words.intersection(content_word_set)
    if matching_words:
        score += len(matching_words) * 3.0 * base_boost
    
    # Term frequency bonus
    for word in matching_words:
        term_freq = content_words.count(word)
        score += min(term_freq * 0.5, 5.0) * base_boost
    
    # Position bonus (content at beginning is often more important)
    for word in query_words:
        if word in content_lower:
            position = content_lower.find(word)
            if position < len(content_lower) * 0.3:  # First 30%
                score += 2.0 * base_boost
            elif position < len(content_lower) * 0.6:  # First 60%
                score += 1.0 * base_boost
    
    # Section heading bonus
    if any(marker in content_lower for marker in [":", " - ", "—", "#", "heading", "title", "chapter"]):
        score += 2.0 * base_boost
    
    # Context type bonus
    tech_keywords = ["navigation", "browser", "driver", "selenium", "webdriver", "click", "open", "tab", "window", "automation"]
    if any(keyword in content_lower for keyword in tech_keywords):
        score += 3.0 * base_boost
    
    # Length bonus (longer content often has more context)
    length_bonus = min(len(content) / 1000, 2.0)  # Cap at 2.0
    score += length_bonus
    
    return min(score, 50.0)  # Cap score

def hybrid_retrieval(query: str, top_k: int = 15) -> List[Dict[str, Any]]:
    """Enhanced hybrid retrieval with better tolerance for spelling variations"""
    
    # Get semantic search results
    semantic_results = semantic_search(query, top_k * 3)
    
    # If we have vector search available, combine results
    vector_results = []
    if pinecone_index is not None and embedding_model is not None:
        try:
            query_embedding = generate_embedding(query)
            
            vector_response = pinecone_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Map vector results to documents
            for match in vector_response['matches']:
                chunk_id = match['id']
                if documents_collection is not None:
                    doc = documents_collection.find_one({"chunk_id": chunk_id})
                    if doc:
                        doc['vector_score'] = match['score']
                        vector_results.append(doc)
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
    
    # Combine and deduplicate results
    all_results = []
    seen_chunks = set()
    
    # Add vector results first (usually more semantically relevant)
    for result in vector_results:
        chunk_id = result["chunk_id"]
        if chunk_id not in seen_chunks:
            seen_chunks.add(chunk_id)
            # Combine scores if available
            if 'score' in result and 'vector_score' in result:
                result['combined_score'] = (result['score'] * 0.4 + result['vector_score'] * 0.6)
            elif 'vector_score' in result:
                result['combined_score'] = result['vector_score']
            all_results.append(result)
    
    # Add semantic results that weren't included
    for result in semantic_results:
        chunk_id = result["chunk_id"]
        if chunk_id not in seen_chunks:
            seen_chunks.add(chunk_id)
            result['combined_score'] = result.get('score', 0)
            all_results.append(result)
    
    # Sort by combined score
    all_results.sort(key=lambda x: x.get('combined_score', x.get('score', 0)), reverse=True)
    
    # Re-rank with cross-encoder if available
    if cross_encoder is not None and all_results:
        try:
            pairs = [(query, doc["content"][:2000]) for doc in all_results[:top_k*2]]
            rerank_scores = cross_encoder.predict(pairs)
            
            for i, doc in enumerate(all_results[:top_k*2]):
                doc['rerank_score'] = float(rerank_scores[i])
            
            all_results.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        except Exception as e:
            logger.warning(f"Re-ranking failed: {e}")
    
    # Be more lenient with relevance threshold
    filtered_results = []
    for result in all_results:
        score = result.get('rerank_score', result.get('combined_score', result.get('score', 0)))
        if score > 0.1:  # Much lower threshold to catch more variations
            filtered_results.append(result)
    
    logger.info(f"🎯 Hybrid retrieval found {len(filtered_results)} relevant chunks")
    return filtered_results[:top_k]

def question_aware_retrieval(query: str, question_type: Dict[str, Any], top_k: int = 15) -> List[Dict[str, Any]]:
    """Enhanced retrieval that considers question type and handles variations"""
    
    # Use processed query for retrieval if available
    retrieval_query = question_type.get('processed_query', query)
    
    # Base retrieval with the query
    results = hybrid_retrieval(retrieval_query, top_k * 2)
    
    # If we have few results, try with the original query as fallback
    if len(results) < 3 and question_type.get('processed_query') != query:
        logger.info("🔄 Few results found, trying with original query...")
        additional_results = hybrid_retrieval(question_type['original_query'], top_k)
        # Merge results
        seen_chunks = set(r["chunk_id"] for r in results)
        for result in additional_results:
            if result["chunk_id"] not in seen_chunks:
                results.append(result)
                seen_chunks.add(result["chunk_id"])
    
    if not results:
        return []
    
    # Question-type specific boosting with enhanced matching
    boosted_results = []
    for result in results:
        content = result["content"].lower()
        score = result.get('rerank_score', result.get('combined_score', result.get('score', 0.5)))
        
        # Enhanced content analysis
        normalized_content = normalize_text(content)
        
        # Boost based on question type with fuzzy matching
        if question_type["requires_person_info"]:
            if any(keyword in normalized_content for keyword in ["student", "name", "person", "individual", "education", "contact"]):
                score *= 2.0
            if "student name" in normalized_content or "name:" in normalized_content or "email" in normalized_content:
                score *= 3.0
                
        elif question_type["requires_contact"]:
            if any(keyword in normalized_content for keyword in ["email", "phone", "mobile", "contact", "@", "gmail"]):
                score *= 3.0
                
        elif question_type["requires_skills"]:
            if any(keyword in normalized_content for keyword in ["skills", "technologies", "languages", "frameworks", "tools"]):
                score *= 2.5
            if any(keyword in normalized_content for keyword in ["python", "java", "javascript", "html", "css", "react", "node"]):
                score *= 2.0
                
        elif question_type["requires_projects"]:
            if any(keyword in normalized_content for keyword in ["projects", "project", "built", "developed", "created"]):
                score *= 3.0
                
        elif question_type["requires_education"]:
            if any(keyword in normalized_content for keyword in ["education", "degree", "college", "university", "institute"]):
                score *= 2.5
                
        elif question_type["requires_technical"]:
            if any(keyword in normalized_content for keyword in ["code", "program", "script", "function", "method", "class", "variable"]):
                score *= 2.0
                
        elif question_type["requires_navigation"]:
            if any(keyword in normalized_content for keyword in ["navigate", "navigation", "browser", "url", "website", "webpage"]):
                score *= 3.0
            if any(keyword in normalized_content for keyword in ["selenium", "webdriver", "driver", "browser", "click", "open"]):
                score *= 2.5
                
        result["question_aware_score"] = score
        boosted_results.append(result)
    
    # Sort by boosted score
    boosted_results.sort(key=lambda x: x.get("question_aware_score", 0), reverse=True)
    
    logger.info(f"🎯 Question-aware boosting applied, returning {len(boosted_results[:top_k])} results")
    return boosted_results[:top_k]

# Answer Generation Functions - IMPROVED VERSION
async def generate_comprehensive_answer(query: str, context_chunks: List[Dict[str, Any]], question_type: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive answer using Gemini API with retrieved context"""
    
    if not context_chunks:
        return await generate_fallback_answer(query)
    
    try:
        # Prepare context for Gemini - include more context for better answers
        context_text = ""
        citation_map = {}
        
        for i, chunk in enumerate(context_chunks[:10]):  # Use top 10 chunks for more comprehensive context
            source_id = f"S{i+1}"
            context_text += f"[{source_id}]: {chunk['content']}\n\n"
            citation_map[source_id] = {
                "chunk_id": chunk["chunk_id"],
                "section": chunk.get("section", "Unknown"),
                "page_number": chunk.get("page_number", 1),
                "score": chunk.get('rerank_score', chunk.get('combined_score', 0.7))
            }
        
        # Create enhanced Gemini prompt with spelling tolerance instructions
        system_prompt = f"""You are an expert educational assistant. Answer the user's question using ONLY the provided context from course documents.

CONTEXT DOCUMENTS:
{context_text}

USER QUESTION: {query}

CRITICAL INSTRUCTIONS:
1. ANSWER STRICTLY AND ONLY USING THE INFORMATION PROVIDED IN THE CONTEXT ABOVE
2. If the answer cannot be found in the context, say "I couldn't find specific information about this in the course materials."
3. Be specific, accurate, and detailed in your answer - provide complete explanations
4. Include relevant details, examples, or explanations from the context
5. If mentioning specific information, reference the source using the format [S1], [S2], etc.
6. Do not make up any information not present in the context
7. If the context contains multiple relevant points, include all of them
8. Structure your answer to be clear and helpful
9. Provide complete code examples or explanations when they appear in the context
10. If the query seems to have spelling variations (like 'navigations' vs 'navigation'), interpret it based on the context
11. Be tolerant of minor spelling differences and focus on semantic meaning

ANSWER:"""
        
        # Use Gemini to generate answer
        response = await gemini_chat_completion(system_prompt, max_tokens=2500)
        
        if "error" in response:
            logger.warning(f"Gemini API failed, using enhanced fallback: {response['error']}")
            return generate_enhanced_structured_answer(query, context_chunks, question_type)
        
        answer_text = response["choices"][0]["message"]["content"]
        
        # Extract citations from answer text
        citations = extract_citations_from_answer(answer_text, citation_map)
        
        # If no citations found but we have context, add the top chunks as citations
        if not citations and context_chunks:
            citations = [{
                "source_id": chunk["chunk_id"],
                "span": f"pg{chunk.get('page_number', 1)}",
                "confidence": chunk.get('rerank_score', chunk.get('combined_score', 0.7)),
                "section": chunk.get("section", "Unknown"),
                "page_number": chunk.get("page_number", 1)
            } for chunk in context_chunks[:3]]
        
        confidence = calculate_enhanced_confidence(context_chunks, query, answer_text)
        
        return {
            "answer": answer_text,
            "citations": citations,
            "confidence": confidence
        }
        
    except Exception as e:
        logger.error(f"Error in Gemini answer generation: {e}")
        # Fallback to enhanced structured answer
        return generate_enhanced_structured_answer(query, context_chunks, question_type)

def extract_citations_from_answer(answer: str, citation_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract citations from answer text based on [S1], [S2] markers"""
    citations = []
    
    # Find all citation markers in the answer
    citation_pattern = r'\[S(\d+)\]'
    matches = re.findall(citation_pattern, answer)
    
    for match in matches:
        source_id = f"S{match}"
        if source_id in citation_map:
            chunk_info = citation_map[source_id]
            citations.append({
                "source_id": chunk_info["chunk_id"],
                "span": f"pg{chunk_info['page_number']}",
                "confidence": chunk_info["score"],
                "section": chunk_info["section"],
                "page_number": chunk_info["page_number"]
            })
    
    # Remove duplicates
    unique_citations = []
    seen_ids = set()
    for citation in citations:
        if citation["source_id"] not in seen_ids:
            unique_citations.append(citation)
            seen_ids.add(citation["source_id"])
    
    return unique_citations

async def generate_fallback_answer(query: str) -> Dict[str, Any]:
    """Generate helpful fallback answer when no specific context is found"""
    # Try a broader search with relaxed criteria
    try:
        broader_results = semantic_search(query, top_k=25)
        if broader_results:
            # Use the broader results to generate an answer
            return generate_enhanced_structured_answer(query, broader_results[:8], {"type": "general"})
    except Exception as e:
        logger.warning(f"Broader search failed: {e}")
    
    return {
        "answer": "I couldn't find specific information about this in the uploaded course documents. The documents may not contain information about this specific topic. Please try asking about content that exists in your uploaded materials.",
        "citations": [],
        "confidence": 0.1
    }

def generate_enhanced_structured_answer(query: str, context_chunks: List[Dict[str, Any]], question_type: Dict[str, Any]) -> Dict[str, Any]:
    """Generate structured answer by analyzing content directly with improved logic"""
    
    if not context_chunks:
        return {
            "answer": "I couldn't find specific information about this topic in the course materials.",
            "citations": [],
            "confidence": 0.1
        }
    
    # Extract key information from all chunks - use more chunks
    relevant_content = []
    for chunk in context_chunks[:6]:
        content = chunk["content"]
        # Check if this chunk is relevant to the query using normalized comparison
        normalized_content = normalize_text(content)
        normalized_query = normalize_text(query)
        
        if (any(term in normalized_content for term in normalized_query.split()) or 
            calculate_semantic_similarity(query, content) > 0.3):
            relevant_content.append(content)
    
    if relevant_content:
        # Combine the most relevant parts
        combined_content = "\n\n".join(relevant_content[:4])
        
        # Try to extract a coherent answer
        sentences = re.split(r'[.!?]+', combined_content)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            normalized_sentence = normalize_text(sentence)
            normalized_query = normalize_text(query)
            
            if (any(term in normalized_sentence for term in normalized_query.split()) and 
                len(sentence.strip()) > 20):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            answer = "Based on the course materials:\n\n" + ". ".join(relevant_sentences[:8]) + "."
            confidence = 0.7
        else:
            # Use the beginning of the most relevant chunks
            previews = [chunk["content"][:300] + "..." for chunk in context_chunks[:3]]
            answer = "Here's relevant information from the course materials:\n\n" + "\n\n".join(previews)
            confidence = 0.5
    else:
        answer = "I found some related information in the documents but couldn't extract a specific answer to your question."
        confidence = 0.3
    
    # Generate citations
    citations = [{
        "source_id": chunk["chunk_id"],
        "span": f"pg{chunk.get('page_number', 1)}",
        "confidence": chunk.get('rerank_score', chunk.get('combined_score', 0.6)),
        "section": chunk.get("section", "Unknown"),
        "page_number": chunk.get("page_number", 1)
    } for chunk in context_chunks[:3]]
    
    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence
    }

def calculate_enhanced_confidence(context_chunks: List[Dict[str, Any]], query: str, answer: str) -> float:
    """Calculate confidence score based on multiple factors"""
    if not context_chunks:
        return 0.1
    
    # Base confidence from retrieval scores
    retrieval_scores = [chunk.get('rerank_score', chunk.get('combined_score', chunk.get('score', 0.3))) for chunk in context_chunks]
    avg_retrieval_score = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.3
    
    # Answer length factor (longer answers often more confident)
    answer_length_factor = min(len(answer) / 800, 1.0)
    
    # Specificity factor (answers with specific details are better)
    specificity_keywords = ["specifically", "for example", "including", "such as", "details", "code", "example"]
    specificity_factor = 1.0 if any(keyword in answer.lower() for keyword in specificity_keywords) else 0.7
    
    # Query coverage factor with normalized text
    query_terms = set(normalize_text(query).split())
    answer_terms = set(normalize_text(answer).split())
    coverage_factor = len(query_terms & answer_terms) / len(query_terms) if query_terms else 0.5
    
    confidence = (avg_retrieval_score * 0.6 + answer_length_factor * 0.2 + 
                 specificity_factor * 0.1 + coverage_factor * 0.1)
    
    return min(confidence, 0.95)

# API Routes
@app.get("/")
async def root():
    return {
        "message": "Course Q&A API - Enhanced Gemini Integration", 
        "status": "running",
        "version": "4.7.0",
        "features": [
            "Enhanced Google Gemini AI Integration",
            "Improved Context Retrieval", 
            "Better Answer Relevance",
            "Advanced Citation System",
            "Spelling Variation Tolerance"
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
        "gemini": gemini_model is not None,
        "services_initialized": services_initialized
    }
    
    overall_status = "healthy" if all([status["mongo"], status["embedding_model"], status["gemini"]]) else "degraded"
    if not status["mongo"]:
        overall_status = "critical"
    
    return {
        "status": overall_status, 
        "services": status,
        "version": "4.7.0"
    }

@app.post("/api/v1/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload and process course material file"""
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not check_database_available():
        raise HTTPException(status_code=503, detail="Database service temporarily unavailable")
    
    # Check file size (limit to 50MB)
    max_size = 50 * 1024 * 1024
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > max_size:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 50MB.")
    
    file_content = await file.read()
    
    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="Empty file provided")
    
    filename = file.filename.lower()
    
    try:
        # Extract text based on file type
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
            raise HTTPException(status_code=400, detail="Unsupported file format. Supported formats: PDF, DOCX, CSV, TXT, MD")
        
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="No text content could be extracted from the file")
        
        logger.info(f"📄 Extracted {len(text)} characters from {file.filename}")
        
        # Use improved chunking with larger chunks for better context
        chunks = intelligent_chunking(text, file.filename, chunk_size=1200, overlap=150)
        logger.info(f"📦 Created {len(chunks)} intelligent chunks from {file.filename}")
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No meaningful content chunks could be created from the extracted text")
        
        # Process chunks
        processed_count = 0
        failed_count = 0
        
        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding
                embedding = generate_embedding(chunk["content"])
                
                # Store in MongoDB
                mongo_doc = chunk.copy()
                documents_collection.insert_one(mongo_doc)
                processed_count += 1
                
                # Store in Pinecone if available
                if pinecone_index is not None:
                    try:
                        metadata = {
                            "filename": chunk["filename"],
                            "section": chunk["section"],
                            "content_preview": chunk["content"][:200],
                            "chunk_index": i,
                            "word_count": len(chunk["content"].split())
                        }
                        
                        pinecone_index.upsert(
                            vectors=[{
                                "id": chunk["chunk_id"],
                                "values": embedding,
                                "metadata": metadata
                            }]
                        )
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to store chunk in Pinecone: {e}")
                        failed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
                failed_count += 1
                continue
        
        success_message = f"Successfully processed {file.filename}. Created {processed_count} chunks."
        if failed_count > 0:
            success_message += f" {failed_count} chunks failed to process in Pinecone (but are stored in MongoDB)."
        
        return UploadResponse(
            message=success_message,
            document_id=str(uuid.uuid4()),
            processing_type=processing_type,
            chunks_processed=processed_count,
            chunks_failed=failed_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in file upload: {e}")
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
        logger.error(f"Error getting document count: {e}")
        return DocumentCountResponse(document_count=0)

@app.delete("/api/v1/documents")
async def clear_all_documents():
    """Clear all documents from both MongoDB and Pinecone"""
    try:
        if not check_database_available():
            return {"message": "Database service not available"}
        
        # Get all chunk IDs before deletion
        chunks = list(documents_collection.find({}, {"chunk_id": 1}))
        chunk_ids = [chunk["chunk_id"] for chunk in chunks] if chunks else []
        
        # Delete from MongoDB
        mongo_result = documents_collection.delete_many({})
        
        # Delete from Pinecone if available
        pinecone_deleted = 0
        if chunk_ids and pinecone_index is not None:
            try:
                batch_size = 100
                for i in range(0, len(chunk_ids), batch_size):
                    batch = chunk_ids[i:i + batch_size]
                    pinecone_index.delete(ids=batch)
                    pinecone_deleted += len(batch)
            except Exception as e:
                logger.error(f"Error deleting from Pinecone: {e}")
        
        return {
            "message": f"Successfully cleared {mongo_result.deleted_count} documents from MongoDB and {pinecone_deleted} vectors from Pinecone"
        }
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing documents: {str(e)}")

@app.get("/api/v1/answer", response_model=AnswerResponse)
async def get_answer(
    query: str,
    lang: str = "en",
    top_k: int = 15
):
    """Get comprehensive answer to course-related question"""
    start_time = time.time()
    
    try:
        if not check_database_available():
            raise HTTPException(status_code=503, detail="Database service temporarily unavailable")
        
        doc_count = documents_collection.count_documents({})
        
        if doc_count == 0:
            raise HTTPException(status_code=404, detail="No course materials found. Please upload documents first.")
        
        logger.info(f"🔍 Processing query: '{query}'")
        
        # Analyze question type with enhanced processing
        question_type = analyze_question_type(query)
        logger.info(f"📝 Question type: {question_type['type']}")
        logger.info(f"🔧 Original query: '{query}' -> Processed: '{question_type.get('processed_query', query)}'")
        
        # Perform enhanced question-aware retrieval
        retrieved_chunks = question_aware_retrieval(query, question_type, top_k)
        logger.info(f"📚 Retrieved {len(retrieved_chunks)} relevant chunks")
        
        # Generate answer using enhanced Gemini with context
        answer_data = await generate_comprehensive_answer(query, retrieved_chunks, question_type)
        
        latency_ms = (time.time() - start_time) * 1000
        
        logger.info(f"✅ Answer generated in {latency_ms:.2f}ms with confidence {answer_data['confidence']:.2f}")
        
        return AnswerResponse(
            answer=answer_data["answer"],
            citations=answer_data["citations"],
            confidence=answer_data["confidence"],
            document_count=doc_count,
            relevant_docs=len(retrieved_chunks),
            latency_ms=latency_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
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
        logger.error(f"Error retrieving source: {e}")
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
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)