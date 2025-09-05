# Installation Guide - Enhanced Document Portal

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- Git
- At least one LLM API key (Groq, OpenAI, Google, etc.)

### 1. Clone Repository
```bash
git clone https://github.com/sunnysavita10/document_portal.git
cd document_portal
```

### 2. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Install missing LangChain text splitters (if needed)
pip install langchain-text-splitters==0.3.27

# Install optional OCR dependencies (for image text extraction)
# Windows: Download Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
# Linux: sudo apt-get install tesseract-ocr
# Mac: brew install tesseract
```

### 4. Configuration
```bash
# Copy environment template
cp env.example .env

# Edit .env file with your API keys
# Minimum required: One LLM API key (GROQ_API_KEY recommended for free tier)
```

### 5. Start Application
```bash
# Development mode
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

# Production mode
uvicorn api.main:app --host 0.0.0.0 --port 8080
```

### 6. Verify Installation
```bash
# Run health check
curl http://localhost:8080/health

# Run basic tests (optional)
pytest tests/test_enhanced_features.py::test_health_endpoint -v
```

## 🔧 Detailed Setup

### API Keys Configuration

#### Free Options
```bash
# Groq (Free tier: 30 requests/minute)
GROQ_API_KEY=gsk_your_groq_key_here

# Google Gemini (Free tier available)
GOOGLE_API_KEY=your_google_api_key_here
```

#### Paid Options
```bash
# OpenAI (Most reliable)
OPENAI_API_KEY=sk-your_openai_key_here

# Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-your_claude_key_here
```

### Optional Services

#### Redis (Recommended for Production)
```bash
# Install Redis
# Windows: Download from https://redis.io/download
# Linux: sudo apt-get install redis-server
# Mac: brew install redis

# Start Redis
redis-server

# Configure in .env
REDIS_URL=redis-cli -h 127.0.0.1 -p 6379
```

#### Database (Optional)
```bash
# PostgreSQL
DATABASE_URL=postgresql://user:password@localhost:5432/document_portal

# MongoDB
MONGODB_URL=mongodb://localhost:27017/document_portal
```

## 📁 Directory Structure
```
document_portal/
├── api/                    # FastAPI application
│   └── main.py            # Main API endpoints
├── src/                   # Core modules
│   ├── document_analyzer/ # Document analysis
│   ├── document_chat/     # Chat functionality
│   ├── document_compare/  # Document comparison
│   └── document_ingestion/# Document processing
├── utils/                 # Enhanced utilities
│   ├── caching.py        # Multi-tier caching
│   ├── enhanced_document_loaders.py # Multi-format support
│   ├── evaluation.py     # DeepEval integration
│   ├── memory_manager.py # Chat memory
│   └── token_counter.py  # Cost analysis
├── tests/                # Comprehensive test suite
├── model/                # Pydantic models
├── static/               # Static files
├── templates/            # HTML templates
└── config/               # Configuration files
```

## 🧪 Testing

### Run All Tests
```bash
# Full test suite
pytest tests/ -v

# Specific test categories
pytest tests/ -m "unit" -v          # Unit tests only
pytest tests/ -m "integration" -v   # Integration tests only
pytest tests/ -m "api" -v           # API tests only
```

### Test Coverage
```bash
# Generate coverage report
pytest tests/ --cov=src --cov=utils --cov=api --cov-report=html

# View coverage report
# Open htmlcov/index.html in browser
```

## 🚨 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Missing langchain-text-splitters
pip install langchain-text-splitters==0.3.27

# Missing deepeval
pip install deepeval==0.21.73

# Missing redis
pip install redis==5.0.1
```

#### API Key Issues
```bash
# Verify API key format
# Groq: starts with 'gsk_'
# OpenAI: starts with 'sk-'
# Google: alphanumeric string

# Test API key
curl -H "Authorization: Bearer $GROQ_API_KEY" https://api.groq.com/openai/v1/models
```

#### Memory Issues
```bash
# Reduce cache size in .env
MEMORY_CACHE_SIZE=100

# Use disk cache only
REDIS_URL=  # Leave empty to disable Redis
```

#### Port Conflicts
```bash
# Use different port
uvicorn api.main:app --port 8081

# Check port usage
netstat -an | findstr :8080  # Windows
lsof -i :8080               # Linux/Mac
```

### Performance Optimization

#### For Large Documents
```bash
# Increase memory limits
export MEMORY_CACHE_SIZE=2000

# Enable Redis caching
REDIS_URL=redis://localhost:6379

# Use chunking for large files
MAX_TOKENS_PER_REQUEST=2000
```

#### For High Traffic
```bash
# Enable all caching layers
REDIS_URL=redis://localhost:6379
CACHE_TTL_EMBEDDINGS=86400
CACHE_TTL_RESPONSES=3600

# Use production ASGI server
pip install gunicorn
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 🔒 Security Setup

### Environment Security
```bash
# Never commit .env file
echo ".env" >> .gitignore

# Use environment-specific configs
cp .env .env.production
cp .env .env.development
```

### API Security
```bash
# Enable CORS for specific domains
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# Set rate limits
RATE_LIMIT_PER_MINUTE=60
```

## 📊 Monitoring Setup

### Enable Logging
```bash
# Create logs directory
mkdir -p logs

# Configure log level
LOG_LEVEL=INFO

# Enable token tracking
ENABLE_TOKEN_TRACKING=true
TOKEN_LOG_FILE=logs/token_usage.jsonl
```

### Health Monitoring
```bash
# Application health
curl http://localhost:8080/health

# Detailed metrics
curl http://localhost:8080/health/detailed

# Cache statistics
curl http://localhost:8080/analytics/cache-stats

# Token usage
curl http://localhost:8080/analytics/token-usage
```

## 🎯 Next Steps

1. **Configure API Keys**: Add at least one LLM API key to `.env`
2. **Test Basic Functionality**: Upload a document and try analysis
3. **Enable Caching**: Set up Redis for better performance
4. **Run Tests**: Validate installation with test suite
5. **Deploy**: Follow `DEPLOYMENT.md` for production setup

## 📞 Support

- **Documentation**: Check `README.md` for feature details
- **Deployment**: See `DEPLOYMENT.md` for production setup
- **Issues**: Create GitHub issues for bugs or feature requests
- **API Reference**: Visit `/docs` endpoint when server is running
