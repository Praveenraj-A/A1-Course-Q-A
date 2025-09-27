import os
import json
import pymongo
import requests
from fastapi import FastAPI, UploadFile, Form, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
import traceback
import numpy as np
import hashlib
import math
import uuid
from datetime import datetime
import PyPDF2
import csv
import io
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import CrossEncoder
import pickle
import sqlite3

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt')
    except:
        print("⚠️ Could not download NLTK data")

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "courseqa")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "documents")
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

print("🔧 Environment Variables Loaded:")

# Setup cross-encoder for re-ranking
try:
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print("✅ Cross-encoder initialized successfully")
except Exception as e:
    print(f"❌ Cross-encoder initialization failed: {e}")
    cross_encoder = None

# Setup storage
client = None
db = None
collection = None
use_local_storage = False
local_storage_file = "local_documents.db"

class LocalStorage:
    def __init__(self, db_file="local_documents.db"):
        self.db_file = db_file
        self.init_db()
    
    def init_db(self):
        """Initialize SQLite database for local storage"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT,
                    chunk_id INTEGER,
                    text TEXT,
                    embedding BLOB,
                    bm25_terms TEXT,
                    source TEXT,
                    section TEXT,
                    metadata TEXT,
                    page_number INTEGER,
                    created_at TEXT
                )
            ''')
            conn.commit()
            conn.close()
            print("✅ Local SQLite storage initialized")
        except Exception as e:
            print(f"❌ Local storage initialization failed: {e}")
    
    def insert_many(self, documents):
        """Insert multiple documents"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            inserted_ids = []
            
            for doc in documents:
                doc_id = str(uuid.uuid4())
                cursor.execute('''
                    INSERT OR REPLACE INTO documents 
                    (id, doc_id, chunk_id, text, embedding, bm25_terms, source, section, metadata, page_number, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    doc_id,
                    doc.get('doc_id', ''),
                    doc.get('chunk_id', 0),
                    doc.get('text', ''),
                    pickle.dumps(doc.get('embedding', [])),
                    doc.get('bm25_terms', ''),
                    doc.get('source', ''),
                    doc.get('section', ''),
                    json.dumps(doc.get('metadata', {})),
                    doc.get('page_number', 0),
                    datetime.utcnow().isoformat()
                ))
                inserted_ids.append(doc_id)
            
            conn.commit()
            conn.close()
            return type('obj', (object,), {'inserted_ids': inserted_ids})
        except Exception as e:
            print(f"❌ Local storage insert failed: {e}")
            return type('obj', (object,), {'inserted_ids': []})
    
    def find(self, query=None, limit=10):
        """Find documents"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM documents LIMIT ?', (limit,))
            
            rows = cursor.fetchall()
            documents = []
            for row in rows:
                documents.append({
                    '_id': row[0],
                    'doc_id': row[1],
                    'chunk_id': row[2],
                    'text': row[3],
                    'embedding': pickle.loads(row[4]) if row[4] else [],
                    'bm25_terms': row[5],
                    'source': row[6],
                    'section': row[7],
                    'metadata': json.loads(row[8]) if row[8] else {},
                    'page_number': row[9],
                    'created_at': row[10]
                })
            
            conn.close()
            return documents
        except Exception as e:
            print(f"❌ Local storage find failed: {e}")
            return []
    
    def delete_many(self, query=None):
        """Delete all documents"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM documents')
            conn.commit()
            deleted_count = cursor.rowcount
            conn.close()
            return type('obj', (object,), {'deleted_count': deleted_count})
        except Exception as e:
            print(f"❌ Local storage delete failed: {e}")
            return type('obj', (object,), {'deleted_count': 0})
    
    def count_documents(self, query=None):
        """Count documents"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM documents')
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            print(f"❌ Local storage count failed: {e}")
            return 0

# Try MongoDB connection
try:
    if MONGO_URI:
        print("🔗 Attempting to connect to MongoDB...")
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("✅ MongoDB connected successfully")
        
        db = client[MONGO_DB]
        collection = db[MONGO_COLLECTION]
        use_local_storage = False
        
        # Create basic indexes (removed vector index creation as it requires special setup)
        try:
            collection.create_index([("bm25_terms", "text")])
            print("✅ Text index created successfully")
        except Exception as e:
            print(f"⚠️ Text index creation warning: {e}")
        
    else:
        raise ValueError("MONGO_URI not provided")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    print("🔄 Falling back to local storage...")
    use_local_storage = True
    local_storage = LocalStorage(local_storage_file)
    collection = local_storage

# Hugging Face configuration
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
print("✅ Hugging Face configured successfully")

app = FastAPI(title="Course Q&A Chatbot", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_embedding(text: str, dimensions: int = 384) -> List[float]:
    """Create embeddings using a simple hash-based approach"""
    try:
        text = text.lower().strip()
        if not text:
            return [0.0] * dimensions
        
        # Use hash-based deterministic embedding
        text_hash = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        np.random.seed(text_hash)
        
        # Create embedding with better semantic properties
        words = text.split()
        embedding = np.zeros(dimensions)
        
        for word in words:
            word_hash = hash(word) % (2**32 - 1)
            np.random.seed(word_hash)
            word_embedding = np.random.randn(dimensions)
            embedding += word_embedding
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.tolist()
        
    except Exception as e:
        print(f"Embedding error: {e}")
        return [0.0] * dimensions

def semantic_chunking(text: str, source: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    """Chunk text by semantic boundaries (headings, paragraphs) for course materials"""
    chunks = []
    
    try:
        # Split by major headings (common in course materials)
        heading_pattern = r'\n#+\s+.+|\n\*+\s+.+|\n[A-Z][A-Z\s]+\n|\n\d+\.\s+.+|\n•\s+.+|\n\b(?:Chapter|Unit|Module|Section)\b'
        sections = re.split(heading_pattern, text)
        headings = re.findall(heading_pattern, text)
        
        current_section = "introduction"
        chunk_id = 0
        page_number = 1
        
        for i, section in enumerate(sections):
            if i < len(headings):
                current_section = headings[i].strip()[:100]  # Limit section name length
            
            # Split section by paragraphs
            paragraphs = [p.strip() for p in section.split('\n\n') if p.strip()]
            
            for paragraph in paragraphs:
                # If paragraph is too long, split by sentences
                if len(paragraph) > chunk_size:
                    sentences = sent_tokenize(paragraph)
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) <= chunk_size:
                            current_chunk += " " + sentence
                        else:
                            if current_chunk.strip():
                                chunks.append(create_chunk(current_chunk.strip(), source, current_section, chunk_id, page_number))
                                chunk_id += 1
                            current_chunk = sentence
                    
                    if current_chunk.strip():
                        chunks.append(create_chunk(current_chunk.strip(), source, current_section, chunk_id, page_number))
                        chunk_id += 1
                else:
                    if paragraph.strip():
                        chunks.append(create_chunk(paragraph, source, current_section, chunk_id, page_number))
                        chunk_id += 1
            
            # Increment page number for each major section (simulated)
            page_number += 1
        
        print(f"✅ Created {len(chunks)} semantic chunks for course materials")
        return chunks
        
    except Exception as e:
        print(f"Semantic chunking error: {e}")
        # Fallback to simple chunking
        words = text.split()
        return [create_chunk(' '.join(words[i:i + chunk_size]), source, "content", i, 1) 
                for i in range(0, len(words), chunk_size - overlap) 
                if i < len(words)]

def create_chunk(text: str, source: str, section: str, chunk_id: int, page_number: int) -> Dict[str, Any]:
    """Create a document chunk with proper metadata for hybrid search"""
    # Extract BM25 terms
    bm25_terms = ' '.join(re.findall(r'\b[a-zA-Z]{3,}\b', text.lower()))
    
    return {
        "doc_id": str(uuid.uuid4()),
        "chunk_id": chunk_id,
        "text": text.strip(),
        "embedding": get_embedding(text.strip()),
        "bm25_terms": bm25_terms,
        "source": source,
        "section": section,
        "page_number": page_number,
        "metadata": {
            "chunk_size": len(text),
            "word_count": len(text.split()),
            "section": section,
            "page_number": page_number,
            "created_at": datetime.utcnow().isoformat()
        },
        "created_at": datetime.utcnow()
    }

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF with page preservation"""
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            
            if page_text.strip():
                # Clean up but preserve structure
                page_text = re.sub(r'\s+', ' ', page_text)
                text += f"--- Page {page_num + 1} ---\n{page_text}\n\n"
        
        return text if text.strip() else ""
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF extraction error: {str(e)}")

def query_huggingface(prompt: str, max_length: int = 1500) -> str:
    """Query Hugging Face API for text generation"""
    if not HUGGING_FACE_API_KEY:
        raise ValueError("Hugging Face API key not available")
    
    headers = {
        "Authorization": f"Bearer {HUGGING_FACE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_length,
            "temperature": 0.1,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            generated_text = result[0].get('generated_text', '')
            return generated_text.strip()
        else:
            raise ValueError("Unexpected response format from Hugging Face API")
            
    except requests.exceptions.Timeout:
        raise Exception("Hugging Face API request timed out")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Hugging Face API request failed: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing Hugging Face response: {str(e)}")

def hybrid_retrieval_with_reranker(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Hybrid retrieval with BM25 + vector search + cross-encoder re-ranking"""
    if collection is None:
        return []
    
    try:
        # Get all documents for local storage, or use database search for MongoDB
        if use_local_storage:
            all_docs = collection.find({}, limit=1000)  # Increased limit for better retrieval
        else:
            # For MongoDB, we'd use proper search - here we simulate with find
            all_docs = list(collection.find({}).limit(1000))
        
        if not all_docs:
            return []
        
        # Vector similarity search
        query_embedding = get_embedding(query)
        vector_results = []
        
        for doc in all_docs:
            doc_embedding = doc.get('embedding', [])
            if doc_embedding and query_embedding:
                try:
                    # Cosine similarity
                    dot_product = np.dot(doc_embedding, query_embedding)
                    norm_a = np.linalg.norm(doc_embedding)
                    norm_b = np.linalg.norm(query_embedding)
                    if norm_a > 0 and norm_b > 0:
                        similarity = dot_product / (norm_a * norm_b)
                        vector_results.append((doc, similarity))
                except:
                    continue
        
        # Sort by vector similarity
        vector_results.sort(key=lambda x: x[1], reverse=True)
        vector_docs = [doc for doc, score in vector_results[:top_k*3]]  # Increased candidate pool
        
        # BM25-like keyword search
        query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
        bm25_results = []
        
        for doc in all_docs:
            doc_text = doc.get('text', '').lower()
            doc_keywords = doc.get('bm25_terms', '').lower()
            
            # Score based on keyword matches
            text_matches = sum(1 for word in query_words if word in doc_text)
            keyword_matches = sum(1 for word in query_words if word in doc_keywords)
            bm25_score = (text_matches * 0.7 + keyword_matches * 0.3) / len(query_words) if query_words else 0
            
            if bm25_score > 0:
                bm25_results.append((doc, bm25_score))
        
        # Sort by BM25 score
        bm25_results.sort(key=lambda x: x[1], reverse=True)
        bm25_docs = [doc for doc, score in bm25_results[:top_k*3]]  # Increased candidate pool
        
        # Combine and deduplicate
        all_candidates = []
        seen_ids = set()
        
        for doc in vector_docs + bm25_docs:
            doc_id = doc.get('_id') or doc.get('doc_id')
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_candidates.append(doc)
        
        # Cross-encoder re-ranking
        if cross_encoder and all_candidates:
            try:
                # Prepare pairs for cross-encoder
                pairs = [(query, doc.get('text', '')) for doc in all_candidates]
                cross_scores = cross_encoder.predict(pairs)
                
                # Add scores to documents
                scored_docs = []
                for doc, score in zip(all_candidates, cross_scores):
                    doc['cross_encoder_score'] = float(score)
                    doc['confidence'] = min(max(float(score), 0), 1.0)  # Ensure 0-1 range
                    scored_docs.append(doc)
                
                # Sort by cross-encoder score
                scored_docs.sort(key=lambda x: x.get('cross_encoder_score', 0), reverse=True)
                final_results = scored_docs[:top_k]
            except Exception as e:
                print(f"Cross-encoder error, using fallback: {e}")
                # Fallback to vector scores
                for doc in all_candidates:
                    vector_score = next((score for d, score in vector_results if (d.get('_id') == doc.get('_id') or d.get('doc_id') == doc.get('doc_id'))), 0)
                    doc['confidence'] = min(vector_score, 1.0)
                
                all_candidates.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                final_results = all_candidates[:top_k]
        else:
            # Fallback: use vector scores
            for doc in all_candidates:
                vector_score = next((score for d, score in vector_results if (d.get('_id') == doc.get('_id') or d.get('doc_id') == doc.get('doc_id'))), 0)
                doc['confidence'] = min(vector_score, 1.0)
            
            all_candidates.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            final_results = all_candidates[:top_k]
        
        # Add source IDs and spans
        for i, doc in enumerate(final_results):
            doc['source_id'] = f"S{i+1}"
            doc['span'] = f"pp{doc.get('page_number', 1)}"
        
        print(f"🔍 Hybrid retrieval found {len(final_results)} results")
        return final_results
        
    except Exception as e:
        print(f"Hybrid retrieval error: {e}")
        traceback.print_exc()
        return []

def generate_grounded_answer(query: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate answers grounded in the retrieved context with proper citations"""
    try:
        if not context_docs:
            return {
                "answer": "I couldn't find specific information in the course materials to answer your question. Please try rephrasing or asking about different topics covered in the syllabus.",
                "citations": [],
                "confidence": 0.0
            }
        
        # Prepare context with proper source identifiers and spans
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            source_id = f"S{i}"
            page_num = doc.get('page_number', 1)
            context_parts.append(f"[{source_id}:pp{page_num}] {doc['text']}")
        
        context = "\n\n".join(context_parts)
        
        # System prompt for grounded answering
        prompt = f"""<s>[INST] You are a helpful teaching assistant answering questions based strictly on the provided course materials.

IMPORTANT GUIDELINES:
1. Answer ONLY using information from the provided context
2. Be precise, factual, and educational
3. Include citations like [S1:pp3], [S2:pp5] for each piece of information
4. If the context doesn't contain relevant information, say so clearly
5. Structure your answer clearly with bullet points or numbered lists when appropriate
6. Keep the answer concise but comprehensive (200-400 words)
7. Reference specific page numbers when available
8. Extract and organize information clearly - don't just copy text verbatim

COURSE MATERIALS:
{context}

QUESTION: {query}

Please provide a well-structured answer with proper citations: [/INST]"""

        try:
            if HUGGING_FACE_API_KEY:
                answer = query_huggingface(prompt)
                
                # Validate that we got a proper response
                if not answer or len(answer) < 50:
                    raise ValueError("Hugging Face returned an empty or very short response")
                    
            else:
                raise ValueError("No Hugging Face API key")
                
        except Exception as e:
            print(f"Hugging Face error: {e}")
            # Use intelligent fallback answer
            answer = generate_intelligent_fallback_answer(query, context_docs)
        
        # Extract citations and create spans
        citations = []
        citation_pattern = r'\[S(\d+):pp(\d+)\]'
        used_sources = set()
        
        # Find all citations in the answer
        for match in re.finditer(citation_pattern, answer):
            source_num, page_num = match.groups()
            used_sources.add((source_num, page_num))
        
        for source_num, page_num in used_sources:
            doc_index = int(source_num) - 1
            if doc_index < len(context_docs):
                doc = context_docs[doc_index]
                citations.append({
                    "source_id": f"S{source_num}",
                    "span": f"pp{page_num}",
                    "text": doc['text'][:300] + "..." if len(doc['text']) > 300 else doc['text'],
                    "confidence": doc.get('confidence', 0.7),
                    "section": doc.get('section', 'content'),
                    "page_number": int(page_num)
                })
        
        overall_confidence = max([doc.get('confidence', 0) for doc in context_docs]) if context_docs else 0.0
        
        print(f"✅ Generated grounded answer with {len(citations)} citations")
        return {
            "answer": answer,
            "citations": citations,
            "confidence": overall_confidence
        }
        
    except Exception as e:
        print(f"Answer generation error: {e}")
        traceback.print_exc()
        return {
            "answer": "I apologize, but I encountered an error while processing your question. Please try again with a different query.",
            "citations": [],
            "confidence": 0.0
        }

def generate_intelligent_fallback_answer(query: str, context_docs: List[Dict[str, Any]]) -> str:
    """Generate a proper fallback answer when Hugging Face fails"""
    if not context_docs:
        return "I couldn't find relevant information in the course materials to answer your question."
    
    # Analyze the query type and generate appropriate response
    query_lower = query.lower()
    
    # Common question patterns
    if any(word in query_lower for word in ['skill', 'ability', 'proficient', 'expert']):
        return generate_skills_answer(context_docs)
    elif any(word in query_lower for word in ['education', 'degree', 'school', 'university']):
        return generate_education_answer(context_docs)
    elif any(word in query_lower for word in ['experience', 'work', 'job', 'employment']):
        return generate_experience_answer(context_docs)
    elif any(word in query_lower for word in ['contact', 'email', 'phone', 'address']):
        return generate_contact_answer(context_docs)
    else:
        return generate_general_answer(query, context_docs)

def generate_skills_answer(context_docs: List[Dict[str, Any]]) -> str:
    """Extract skills information from documents"""
    skills_info = []
    
    for i, doc in enumerate(context_docs, 1):
        text = doc['text'].lower()
        page_num = doc.get('page_number', 1)
        
        # Look for skills-related content
        if any(keyword in text for keyword in ['skill', 'proficient', 'expert', 'knowledge', 'ability', 'technical']):
            # Extract relevant sentences
            sentences = re.split(r'[.!?]+', doc['text'])
            skill_sentences = []
            
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in ['skill', 'proficient', 'expert', 'knowledge', 'ability', 'technical']):
                    skill_sentences.append(sentence.strip())
            
            if skill_sentences:
                skills_info.append(f"[S{i}:pp{page_num}] " + " ".join(skill_sentences[:3]))  # Limit to 3 sentences
    
    if skills_info:
        return "Based on the course materials, here are the key skills mentioned:\n\n" + "\n\n".join(skills_info)
    else:
        return "I found information in the materials but couldn't identify specific skills sections. The document appears to contain: " + \
               context_docs[0]['text'][:200] + "..."

def generate_education_answer(context_docs: List[Dict[str, Any]]) -> str:
    """Extract education information from documents"""
    education_info = []
    
    for i, doc in enumerate(context_docs, 1):
        text = doc['text'].lower()
        page_num = doc.get('page_number', 1)
        
        # Look for education-related content
        if any(keyword in text for keyword in ['education', 'degree', 'university', 'college', 'bachelor', 'master', 'phd']):
            # Extract relevant sentences
            sentences = re.split(r'[.!?]+', doc['text'])
            edu_sentences = []
            
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in ['education', 'degree', 'university', 'college', 'bachelor', 'master', 'phd']):
                    edu_sentences.append(sentence.strip())
            
            if edu_sentences:
                education_info.append(f"[S{i}:pp{page_num}] " + " ".join(edu_sentences[:3]))
    
    if education_info:
        return "Based on the course materials, here is the educational background:\n\n" + "\n\n".join(education_info)
    else:
        return "I found information in the materials but couldn't identify specific education details."

def generate_experience_answer(context_docs: List[Dict[str, Any]]) -> str:
    """Extract work experience information from documents"""
    experience_info = []
    
    for i, doc in enumerate(context_docs, 1):
        text = doc['text'].lower()
        page_num = doc.get('page_number', 1)
        
        # Look for experience-related content
        if any(keyword in text for keyword in ['experience', 'work', 'job', 'employment', 'position', 'role']):
            # Extract relevant sentences
            sentences = re.split(r'[.!?]+', doc['text'])
            exp_sentences = []
            
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in ['experience', 'work', 'job', 'employment', 'position', 'role']):
                    exp_sentences.append(sentence.strip())
            
            if exp_sentences:
                experience_info.append(f"[S{i}:pp{page_num}] " + " ".join(exp_sentences[:3]))
    
    if experience_info:
        return "Based on the course materials, here is the work experience information:\n\n" + "\n\n".join(experience_info)
    else:
        return "I found information in the materials but couldn't identify specific work experience details."

def generate_contact_answer(context_docs: List[Dict[str, Any]]) -> str:
    """Extract contact information from documents"""
    contact_info = []
    
    for i, doc in enumerate(context_docs, 1):
        text = doc['text']
        page_num = doc.get('page_number', 1)
        
        # Look for contact information patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        
        emails = re.findall(email_pattern, text)
        phones = re.findall(phone_pattern, text)
        
        if emails or phones:
            contact_details = []
            if emails:
                contact_details.append(f"Email: {emails[0]}")
            if phones:
                contact_details.append(f"Phone: {phones[0]}")
            
            contact_info.append(f"[S{i}:pp{page_num}] " + "; ".join(contact_details))
    
    if contact_info:
        return "Based on the course materials, here is the contact information:\n\n" + "\n\n".join(contact_info)
    else:
        return "I found information in the materials but couldn't identify specific contact details."

def generate_general_answer(query: str, context_docs: List[Dict[str, Any]]) -> str:
    """Generate a general answer for other types of queries"""
    answer_parts = []
    
    for i, doc in enumerate(context_docs[:3], 1):  # Limit to 3 most relevant docs
        page_num = doc.get('page_number', 1)
        # Take first 100 words of the most relevant content
        words = doc['text'].split()[:100]
        preview = ' '.join(words) + ('...' if len(doc['text'].split()) > 100 else '')
        answer_parts.append(f"[S{i}:pp{page_num}] {preview}")
    
    return f"Based on the course materials, here's relevant information for '{query}':\n\n" + "\n\n".join(answer_parts)

@app.get("/")
async def root():
    return {
        "message": "Course Q&A Chatbot - RAG with Hybrid Retrieval", 
        "version": "1.0.0",
        "status": "operational"
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process course materials (syllabus, FAQs, etc.)"""
    if collection is None:
        raise HTTPException(status_code=500, detail="Storage not available")
    
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    try:
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        file_extension = file.filename.split('.')[-1].lower()
        text_data = ""
        
        print(f"📤 Processing {file_extension} file: {file.filename}")
        
        if file_extension == 'pdf':
            text_data = extract_text_from_pdf(content)
        elif file_extension in ['txt', 'md', 'csv']:
            text_data = content.decode('utf-8')
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
        
        if not text_data.strip():
            raise HTTPException(status_code=400, detail="No readable text content found in file")
        
        # Use semantic chunking for course materials
        chunks = semantic_chunking(text_data, file.filename)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract meaningful content from file")
        
        # Clear existing documents before uploading new ones
        try:
            result = collection.delete_many({})
            print(f"🗑️ Cleared {result.deleted_count} existing documents")
        except Exception as e:
            print(f"⚠️ Error clearing documents: {e}")
        
        inserted_count = 0
        if chunks:
            result = collection.insert_many(chunks)
            inserted_count = len(result.inserted_ids) if hasattr(result, 'inserted_ids') else len(chunks)
        
        actual_count = collection.count_documents({})
        
        return {
            "status": "success", 
            "message": f"Course materials processed successfully. Created {inserted_count} semantic chunks.",
            "filename": file.filename,
            "chunks_created": inserted_count,
            "total_documents": actual_count,
            "processing_type": "semantic_chunking"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Upload error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/answer")
async def answer_question(
    query: str = Form(...),
    lang: Optional[str] = Form("en"),
    top_k: Optional[int] = Form(5)
):
    """Main endpoint for answering questions about course materials"""
    if collection is None:
        raise HTTPException(status_code=500, detail="Storage not available")
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    start_time = datetime.utcnow()
    
    try:
        doc_count = collection.count_documents({})
        if doc_count == 0:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "answer": "No course materials found. Please upload syllabus or course documents first.",
                    "document_count": 0
                }
            )
        
        print(f"🧠 Processing query: '{query}' (lang: {lang}, top_k: {top_k})")
        
        # Perform hybrid retrieval with re-ranking
        docs = hybrid_retrieval_with_reranker(query, top_k=top_k)
        
        if not docs:
            return {
                "status": "success",
                "answer": f"I searched through the course materials but couldn't find specific information matching '{query}'. Please try rephrasing your question or ask about different topics covered in the syllabus.",
                "citations": [],
                "relevant_docs": 0,
                "confidence": 0.0,
                "document_count": doc_count
            }
        
        # Generate grounded answer
        result = generate_grounded_answer(query, docs)
        
        # If confidence is very low, adjust the response
        if result["confidence"] < 0.3:
            result["answer"] = "I found some information in the course materials, but it may not directly answer your question. " + result["answer"]
        
        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "status": "success",
            "query": query,
            "answer": result["answer"],
            "citations": result["citations"],
            "relevant_docs": len(docs),
            "confidence": result["confidence"],
            "latency_ms": round(latency_ms, 2),
            "document_count": doc_count
        }
        
    except Exception as e:
        print(f"Answer question error: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "answer": f"Error processing your question: {str(e)}",
                "relevant_docs": 0,
                "confidence": 0.0,
                "document_count": collection.count_documents({}) if collection else 0
            }
        )

@app.get("/source/{source_id}")
async def get_source(source_id: str):
    """Get source document by ID with metadata"""
    if collection is None:
        raise HTTPException(status_code=500, detail="Storage not available")
    
    try:
        # Extract numeric part from source_id (e.g., "S1" -> 1)
        try:
            doc_index = int(source_id[1:]) - 1 if source_id.startswith('S') else -1
        except:
            doc_index = -1
        
        if doc_index < 0:
            raise HTTPException(status_code=400, detail="Invalid source ID format")
        
        # Get all documents and select by index
        if use_local_storage:
            all_docs = collection.find({}, limit=1000)
            all_docs.sort(key=lambda x: x.get('_id', ''))
        else:
            all_docs = list(collection.find({}).limit(1000))
        
        if doc_index >= len(all_docs):
            raise HTTPException(status_code=404, detail="Source not found")
        
        doc = all_docs[doc_index]
        
        return {
            "status": "success",
            "source_id": source_id,
            "text": doc['text'],
            "metadata": doc.get('metadata', {}),
            "source": doc.get('source', 'unknown'),
            "page_number": doc.get('page_number', 1)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error retrieving source: {str(e)}")

@app.post("/feedback")
async def submit_feedback(
    query: str = Form(...),
    answer_id: str = Form(...),
    label: str = Form(..., regex="^(good|bad)$"),
    note: Optional[str] = Form(None)
):
    """Submit feedback for an answer"""
    try:
        # In a production system, store this in a feedback database
        print(f"📝 Feedback received: {label} for answer {answer_id}")
        print(f"Query: {query}")
        print(f"Note: {note}")
        
        return {
            "status": "success",
            "message": "Feedback recorded successfully",
            "feedback_id": str(uuid.uuid4())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recording feedback: {str(e)}")

@app.delete("/documents")
async def clear_all_documents():
    """Clear all documents from the database"""
    if collection is None:
        raise HTTPException(status_code=500, detail="Storage not available")
    
    try:
        result = collection.delete_many({})
        new_count = collection.count_documents({})
        return {
            "status": "success",
            "message": f"Cleared {result.deleted_count} documents.",
            "deleted_count": getattr(result, 'deleted_count', 0),
            "current_count": new_count
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error clearing documents: {str(e)}")

@app.get("/documents/count")
async def get_document_count():
    """Get the count of documents in the database"""
    try:
        if collection is None:
            return {"status": "success", "document_count": 0}
        count = collection.count_documents({})
        return {"status": "success", "document_count": count}
    except Exception as e:
        print(f"Error counting documents: {e}")
        return {"status": "error", "document_count": 0, "error": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        doc_count = collection.count_documents({}) if collection else 0
        return {
            "status": "healthy",
            "document_count": doc_count,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Course Q&A Chatbot on http://0.0.0.0:8000")
    print("📚 Designed for educational content: syllabi, FAQs, course materials")
    print("🤗 Using Hugging Face API for LLM")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")