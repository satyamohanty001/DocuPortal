# Document Portal - Quick Start Guide

## 🚀 Starting the Enhanced Document Portal

### Method 1: Using uvicorn directly
```bash
cd C:\LLMops\Myprojects\Enhancement\document_portal
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

### Method 2: Using Python
```bash
cd C:\LLMops\Myprojects\Enhancement\document_portal
python -c "import uvicorn; uvicorn.run('api.main:app', host='0.0.0.0', port=8080, reload=True)"
```

### Method 3: Direct Python execution
```bash
cd C:\LLMops\Myprojects\Enhancement\document_portal
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

## 🧪 Testing the Application

### 1. After starting the server, run the test script:
```bash
python test_server.py
```

### 2. Open your browser and navigate to:
```
http://localhost:8080
```

## 🎯 Enhanced Features to Test

### Analytics Tab
- **Cache Statistics**: View real-time cache hits, misses, and performance metrics
- **Token Usage**: Monitor API token consumption and cost estimates
- **System Health**: Check system status and active sessions
- **Document Statistics**: View processing metrics and file type distribution

### Document Analysis Tab
- **Multi-format Support**: Upload PDF, DOCX, TXT, MD, XLSX, CSV files
- **Evaluation Metrics**: View quality scores for document analysis
- **Token Tracking**: See token usage for each analysis
- **Caching**: Experience faster responses for repeated analyses

### Chat Tab
- **Memory Management**: Persistent conversation history
- **Source References**: View document sources for answers
- **Quality Scores**: Evaluation metrics for chat responses
- **Session Management**: Multiple conversation sessions

### Document Compare Tab
- **Enhanced Comparison**: Compare documents with detailed analysis
- **Performance Metrics**: View processing time and resource usage

## 🔧 API Endpoints Available

### Health & Monitoring
- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed system metrics

### Analytics
- `GET /analytics/cache-stats` - Cache performance statistics
- `GET /analytics/token-usage` - Token usage and cost breakdown

### Document Processing
- `POST /analyze` - Basic document analysis
- `POST /analyze/enhanced` - Enhanced analysis with caching and metrics
- `POST /compare` - Document comparison

### Chat & RAG
- `POST /chat/index` - Build document index for chat
- `POST /chat/query` - Basic chat query
- `POST /chat/query/enhanced` - Enhanced chat with memory and evaluation

### Session Management
- `GET /sessions/{session_id}/history` - Get conversation history
- `GET /sessions/{session_id}/context` - Get session context
- `POST /sessions/cleanup` - Clean up old sessions

### Utilities
- `DELETE /cache/clear` - Clear all caches
- `POST /evaluation/batch` - Batch evaluate responses

## 🎨 UI Features

### Enhanced Interface
- **Tabbed Navigation**: Easy switching between features
- **Real-time Updates**: Dynamic data fetching and display
- **Responsive Design**: Works on desktop and mobile
- **Interactive Charts**: Visual representation of metrics
- **Progress Indicators**: Loading states and processing feedback

### User Experience Improvements
- **File Drag & Drop**: Easy document uploading
- **Progress Bars**: Visual feedback during processing
- **Error Handling**: User-friendly error messages
- **Keyboard Shortcuts**: Quick navigation and actions
- **Dark/Light Theme**: Automatic theme detection

## 🔍 Testing Checklist

- [ ] Server starts without errors
- [ ] UI loads at http://localhost:8080
- [ ] Analytics tab shows cache and token stats
- [ ] Document upload works in Analysis tab
- [ ] Chat indexing and querying functions
- [ ] Document comparison works
- [ ] All API endpoints respond correctly
- [ ] Error handling works properly
- [ ] Performance metrics are displayed
- [ ] Session management functions correctly

## 🐛 Troubleshooting

### Common Issues
1. **Port 8080 already in use**: Change port or kill existing process
2. **Import errors**: Install missing dependencies with `pip install -r requirements.txt`
3. **API key errors**: Set up environment variables from `env.example`
4. **Cache errors**: Ensure Redis is running (optional) or disable Redis in config

### Performance Tips
1. **Enable Redis**: For better caching performance
2. **Set API Keys**: For full LLM functionality
3. **Adjust Chunk Size**: For optimal document processing
4. **Monitor Memory**: Large documents may require more RAM

## 📊 Expected Performance

### With Caching Enabled
- **First Analysis**: 2-5 seconds
- **Cached Analysis**: 0.1-0.5 seconds
- **Chat Responses**: 1-3 seconds
- **UI Loading**: <1 second

### Supported File Sizes
- **Text Files**: Up to 10MB
- **PDF Files**: Up to 50MB
- **Office Documents**: Up to 25MB
- **Images**: Up to 10MB (with OCR)

Ready to test your enhanced document portal! 🎉
