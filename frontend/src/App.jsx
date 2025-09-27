import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { FiSend, FiPaperclip, FiUser, FiMessageSquare, FiUpload, FiCheck, FiTrash2, FiFile, FiX, FiBook, FiRefreshCw, FiExternalLink, FiDownload, FiSearch } from 'react-icons/fi';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [isFileUploaded, setIsFileUploaded] = useState(false);
  const [documentCount, setDocumentCount] = useState(0);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [activeCitations, setActiveCitations] = useState([]);
  const [suggestedQuestions, setSuggestedQuestions] = useState([]);
  const chatEndRef = useRef(null);

  const suggestedQuestionsList = [
    "What are the key skills mentioned?",
    "What is the educational background?",
    "List the work experience",
    "What projects are included?",
    "What certifications does the person have?",
    "Extract contact information",
    "What are the technical proficiencies?",
    "Summarize the professional experience"
  ];

  const refreshDocumentCount = async () => {
    setIsRefreshing(true);
    try {
      const response = await axios.get('http://localhost:8000/documents/count');
      setDocumentCount(response.data.document_count);
    } catch (error) {
      console.error('Error fetching document count:', error);
    }
    setIsRefreshing(false);
  };

  useEffect(() => {
    refreshDocumentCount();
    const interval = setInterval(refreshDocumentCount, 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    // Show suggested questions when no messages
    if (messages.length === 0 && documentCount > 0) {
      setSuggestedQuestions(suggestedQuestionsList);
    } else {
      setSuggestedQuestions([]);
    }
  }, [messages, documentCount]);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleFileUpload = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await axios.post('http://localhost:8000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 30000
      });

      setIsFileUploaded(true);
      setMessages((prev) => [...prev, {
        text: response.data.message,
        sender: 'system',
        timestamp: new Date().toLocaleTimeString(),
        isFileUpload: true,
        processingType: response.data.processing_type
      }]);
      setSelectedFile(null);
      
      setTimeout(refreshDocumentCount, 1000);
      setTimeout(() => setIsFileUploaded(false), 3000);
    } catch (err) {
      console.error('Upload Error:', err);
      setError(err.response?.data?.detail || 'Failed to upload file. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  const clearAllDocuments = async () => {
    if (!window.confirm('Are you sure you want to clear all documents?')) {
      return;
    }

    try {
      const response = await axios.delete('http://localhost:8000/documents');
      setMessages((prev) => [...prev, {
        text: response.data.message,
        sender: 'system',
        timestamp: new Date().toLocaleTimeString()
      }]);
      refreshDocumentCount();
      setMessages([]);
    } catch (err) {
      console.error('Clear documents error:', err);
      setError('Failed to clear documents. Please try again.');
    }
  };

  const handleSend = async (e, question = null) => {
    const questionText = question || query;
    if (!questionText.trim()) return;

    if (e) e.preventDefault();

    const userMessage = { 
      text: questionText, 
      sender: 'user', 
      timestamp: new Date().toLocaleTimeString(),
      id: Date.now()
    };
    setMessages((prev) => [...prev, userMessage]);
    setQuery('');
    setIsLoading(true);
    setError(null);
    setActiveCitations([]);
    setSuggestedQuestions([]);

    try {
      const formData = new FormData();
      formData.append('query', questionText);

      const response = await axios.post('http://localhost:8000/answer', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 60000
      });

      // Handle both success and error responses from backend
      if (response.data.status === 'error') {
        setError(response.data.answer || 'Sorry, something went wrong. Please try again.');
        return;
      }

      const botMessage = {
        text: response.data.answer,
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString(),
        confidence: response.data.confidence,
        documentCount: response.data.document_count,
        relevantDocs: response.data.relevant_docs,
        citations: response.data.citations || [],
        answerLength: response.data.answer?.length || 0,
        id: Date.now() + 1
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      console.error('API Error:', err);
      if (err.response?.status === 404) {
        setError('No documents found. Please upload course materials first.');
      } else {
        setError('Sorry, something went wrong. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setIsFileUploaded(false);
      setError(null);
    }
  };

  const removeSelectedFile = () => {
    setSelectedFile(null);
    setIsFileUploaded(false);
  };

  const clearChat = () => {
    setMessages([]);
    setError(null);
    setActiveCitations([]);
  };

  const exportChat = () => {
    const chatContent = messages.map(msg => 
      `${msg.sender === 'user' ? 'YOU' : 'ASSISTANT'} (${msg.timestamp}):\n${msg.text}\n${'-'.repeat(50)}`
    ).join('\n\n');
    
    const blob = new Blob([chatContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-export-${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const toggleCitation = async (citation) => {
    // Check if citation is already active
    if (activeCitations.some(c => c.source_id === citation.source_id)) {
      setActiveCitations(prev => prev.filter(c => c.source_id !== citation.source_id));
      return;
    }

    try {
      // Fetch source details from backend
      const response = await axios.get(`http://localhost:8000/source/${citation.source_id}`);
      
      const citationWithDetails = {
        ...citation,
        fullText: response.data.text,
        metadata: response.data.metadata
      };
      
      setActiveCitations(prev => [...prev, citationWithDetails]);
    } catch (err) {
      console.error('Error fetching source details:', err);
      // Fallback to existing citation data if API call fails
      setActiveCitations(prev => [...prev, citation]);
    }
  };

  const formatTextWithCitations = (text) => {
    if (!text) return text;
    
    // Updated regex to match backend citation format [S1:pp3]
    const parts = text.split(/(\[S\d+:pp\d+\])/g);
    
    return parts.map((part, index) => {
      if (part.match(/\[S\d+:pp\d+\]/)) {
        return (
          <mark key={index} className="citation-marker">
            {part}
          </mark>
        );
      }
      return part;
    });
  };

  const handleSuggestedQuestion = (question) => {
    handleSend(null, question);
  };

  return (
    <div className="app-container">
      <div className="sidebar">
        <div className="sidebar-header">
          <FiBook className="logo" />
          <h2>Course Q&A</h2> {/* Updated title to match backend */}
        </div>
        
        <div className="document-stats">
          <div className="stat-item">
            <span className="stat-label">Documents</span>
            <div className="stat-value-container">
              <span className="stat-value">{documentCount}</span>
              <button 
                onClick={refreshDocumentCount} 
                className="refresh-btn"
                disabled={isRefreshing}
              >
                <FiRefreshCw className={isRefreshing ? 'spinning' : ''} />
              </button>
            </div>
          </div>
        </div>

        <div className="sidebar-actions">
          <button 
            className="clear-chat-btn"
            onClick={clearChat}
            disabled={messages.length === 0}
          >
            <FiX /> Clear Chat
          </button>
          
          <button 
            className="export-chat-btn"
            onClick={exportChat}
            disabled={messages.length === 0}
          >
            <FiDownload /> Export Chat
          </button>
          
          <button 
            className="clear-docs-btn"
            onClick={clearAllDocuments}
            disabled={documentCount === 0}
          >
            <FiTrash2 /> Clear Docs
          </button>
        </div>

        <div className="upload-section">
          <h3>Upload Course Material</h3> {/* Updated label */}
          <div className="upload-controls">
            <label htmlFor="file-upload" className="file-upload-label">
              <FiPaperclip />
              Choose File
            </label>
            <input
              id="file-upload"
              type="file"
              onChange={handleFileChange}
              accept=".pdf,.txt,.csv,.md,.doc,.docx"
              style={{ display: 'none' }}
            />
            
            {selectedFile && (
              <div className="file-info">
                <FiFile className="file-icon" />
                <span className="file-name">{selectedFile.name}</span>
                <div className="file-actions">
                  <button 
                    onClick={handleFileUpload}
                    disabled={isUploading}
                    className="upload-btn"
                  >
                    {isUploading ? 'Uploading...' : <><FiUpload /> Upload</>}
                  </button>
                  <button 
                    onClick={removeSelectedFile}
                    className="remove-btn"
                  >
                    <FiX />
                  </button>
                </div>
              </div>
            )}
            
            {isFileUploaded && (
              <div className="upload-success">
                <FiCheck />
                <span>Uploaded successfully!</span>
              </div>
            )}
          </div>
        </div>

        <div className="usage-tips">
          <h4>💡 Course Features</h4> {/* Updated tips */}
          <ul>
            <li>Semantic section-based extraction</li>
            <li>Hybrid retrieval with re-ranking</li>
            <li>Structured answer generation</li>
            <li>Clickable source references</li>
          </ul>
        </div>
      </div>

      <div className="main-content">
        <div className="chat-header">
          <h1>Course Q&A Chatbot</h1> {/* Updated title */}
          <p>Intelligent information extraction from course materials with precise citations</p>
        </div>

        <div className="chat-body">
          {messages.length === 0 ? (
            <div className="welcome-message">
              <FiSearch className="welcome-icon" />
              <h2>Course Material Analysis</h2> {/* Updated welcome message */}
              <p>Upload course materials and ask specific questions to get precise, cited information</p>
              
              {documentCount > 0 && suggestedQuestions.length > 0 && (
                <div className="suggested-questions">
                  <h3>💡 Try asking:</h3>
                  <div className="question-grid">
                    {suggestedQuestions.map((question, index) => (
                      <button
                        key={index}
                        className="question-btn"
                        onClick={() => handleSuggestedQuestion(question)}
                        disabled={isLoading}
                      >
                        {question}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              
              <div className="welcome-features">
                <div className="feature">
                  <FiSearch />
                  <span>Semantic section detection</span>
                </div>
                <div className="feature">
                  <FiMessageSquare />
                  <span>Precise information extraction</span>
                </div>
                <div className="feature">
                  <FiDownload />
                  <span>Citation-based answers</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="messages-container">
              {messages.map((msg) => (
                <div key={msg.id} className={`message ${msg.sender}`}>
                  <div className="message-avatar">
                    {msg.sender === 'user' ? <FiUser /> : <FiMessageSquare />}
                  </div>
                  <div className="message-content">
                    <div className="message-header">
                      <span className="sender-name">
                        {msg.sender === 'user' ? 'You' : 'Teaching Assistant'} {/* Updated bot name */}
                      </span>
                      <span className="message-time">{msg.timestamp}</span>
                    </div>
                    
                    <div className="message-text">
                      {formatTextWithCitations(msg.text)}
                    </div>
                    
                    {msg.sender === 'bot' && msg.citations && msg.citations.length > 0 && (
                      <div className="citations-section">
                        <div className="citations-header">
                          <FiExternalLink />
                          <span>Source Sections:</span>
                        </div>
                        <div className="citations-list">
                          {msg.citations.map((citation, index) => (
                            <button
                              key={index}
                              className={`citation-btn ${activeCitations.some(c => c.source_id === citation.source_id) ? 'active' : ''}`}
                              onClick={() => toggleCitation(citation)}
                            >
                              {citation.source_id} ({citation.span}) {/* Updated to show span instead of section */}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {msg.sender === 'bot' && (
                      <div className="message-meta">
                        <span>Docs: {msg.documentCount}</span>
                        {msg.relevantDocs > 0 && <span>Matches: {msg.relevantDocs}</span>}
                        {msg.confidence > 0 && <span>Confidence: {Math.round(msg.confidence * 100)}%</span>}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              
              {isLoading && (
                <div className="message bot loading">
                  <div className="message-avatar">
                    <FiMessageSquare />
                  </div>
                  <div className="message-content">
                    <div className="thinking">Analyzing course materials...</div> {/* Updated loading text */}
                    <div className="loading-dots">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>
          )}
        </div>

        {activeCitations.length > 0 && (
          <div className="citations-panel">
            <div className="panel-header">
              <h3>Source Details</h3>
              <button onClick={() => setActiveCitations([])} className="close-panel">
                <FiX />
              </button>
            </div>
            <div className="citations-content">
              {activeCitations.map((citation, index) => (
                <div key={index} className="citation-detail">
                  <h4>{citation.source_id} - {citation.span}</h4> {/* Updated to show span */}
                  {citation.section && <div className="citation-section">Section: {citation.section}</div>}
                  <div className="citation-text">
                    {citation.fullText || citation.text} {/* Show full text if available */}
                  </div>
                  <div className="citation-meta">
                    <span>Confidence: {Math.round(citation.confidence * 100)}%</span>
                    {citation.page_number && <span>Page: {citation.page_number}</span>}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {error && (
          <div className="error-message">
            <FiX className="error-icon" />
            <span>{error}</span>
            <button onClick={() => setError(null)} className="error-close">
              <FiX />
            </button>
          </div>
        )}

        <form className="chat-input-form" onSubmit={handleSend}>
          <div className="input-container">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask about course content like 'What are the key topics?' or 'Explain the grading policy'..."
              disabled={isLoading || isUploading}
              className="message-input"
            />
            <button 
              type="submit" 
              disabled={isLoading || isUploading || !query.trim()}
              className="send-button"
            >
              <FiSend />
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default App;