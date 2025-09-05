# Contributing to Enhanced Document Portal

Thank you for your interest in contributing to the Enhanced Document Portal! This guide will help you get started with contributing to this project.

## 🚀 Quick Start for Contributors

### Prerequisites
- Python 3.8+
- Git
- Free API keys (Groq and/or Google Gemini)

### Development Setup

1. **Fork and Clone**
```bash
git clone https://github.com/yourusername/document-portal.git
cd document-portal
git remote add upstream https://github.com/original/document-portal.git
```

2. **Set Up Environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

3. **Configure Environment**
```bash
cp env.example .env
# Add your free API keys to .env
```

4. **Install Pre-commit Hooks**
```bash
pre-commit install
```

## 🎯 How to Contribute

### Types of Contributions Welcome
- 🐛 Bug fixes
- ✨ New features
- 📚 Documentation improvements
- 🧪 Test coverage improvements
- 🎨 UI/UX enhancements
- ⚡ Performance optimizations

### Before You Start
1. Check existing [issues](https://github.com/yourusername/document-portal/issues)
2. Create an issue for new features or major changes
3. Comment on issues you'd like to work on

## 📝 Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Your Changes
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Test Your Changes
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific tests
pytest tests/test_your_feature.py -v

# Test the API server
python test_server.py
```

### 4. Code Quality Checks
```bash
# Format code
black .

# Sort imports
isort .

# Lint code
flake8 .

# Type checking
mypy .

# Run all pre-commit hooks
pre-commit run --all-files
```

### 5. Commit Your Changes
```bash
git add .
git commit -m "feat: add new document format support"
# or
git commit -m "fix: resolve caching issue with large files"
```

**Commit Message Convention:**
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Test additions/modifications
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `style:` Code style changes

### 6. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## 🧪 Testing Guidelines

### Test Structure
```
tests/
├── test_api_integration.py      # API endpoint tests
├── test_enhanced_features.py    # Enhanced features tests
├── test_routes.py              # Route tests
├── conftest.py                 # Test configuration
└── data/                       # Test data files
```

### Writing Tests
```python
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_document_analysis():
    """Test document analysis endpoint."""
    with open("tests/data/sample.pdf", "rb") as f:
        response = client.post(
            "/analyze/enhanced",
            files={"file": ("sample.pdf", f, "application/pdf")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "evaluation_metrics" in data["data"]
    assert "token_usage" in data["data"]
```

### Test Coverage Requirements
- Minimum 80% code coverage
- All new features must have tests
- Critical paths must be thoroughly tested

## 📚 Documentation Guidelines

### Code Documentation
```python
def analyze_document(text: str) -> Dict[str, Any]:
    """
    Analyze document content and extract insights.
    
    Args:
        text: The document text to analyze
        
    Returns:
        Dictionary containing analysis results with keys:
        - summary: Document summary
        - key_points: List of key points
        - sentiment: Sentiment analysis score
        
    Raises:
        DocumentPortalException: If analysis fails
    """
```

### README Updates
- Update feature lists for new capabilities
- Add usage examples for new features
- Update API documentation for new endpoints

## 🎨 UI/Frontend Guidelines

### CSS Standards
- Use existing CSS custom properties
- Follow mobile-first responsive design
- Maintain consistent spacing and typography
- Test across different browsers

### JavaScript Standards
- Use vanilla JavaScript (no frameworks)
- Follow existing naming conventions
- Add error handling for API calls
- Ensure accessibility compliance

## 🔧 API Development Guidelines

### Endpoint Standards
```python
@app.post("/api/new-feature", response_model=APIResponse)
async def new_feature_endpoint(
    request: NewFeatureRequest,
    background_tasks: BackgroundTasks
) -> APIResponse:
    """
    Brief description of the endpoint.
    
    Args:
        request: Request model with validation
        background_tasks: For async processing
        
    Returns:
        Standardized API response
    """
    try:
        # Implementation
        return APIResponse(success=True, data=result)
    except Exception as e:
        log.exception("Feature failed")
        return APIResponse(success=False, error=str(e))
```

### Model Standards
- Use Pydantic V2 models
- Add proper validation
- Include helpful docstrings
- Use appropriate field types

## 🚀 Performance Guidelines

### Caching
- Use appropriate cache levels (memory/disk/Redis)
- Set reasonable TTL values
- Cache expensive operations
- Implement cache invalidation

### Database Operations
- Use connection pooling
- Implement proper indexing
- Avoid N+1 queries
- Use batch operations when possible

### API Optimization
- Implement request/response compression
- Use appropriate HTTP status codes
- Add rate limiting for expensive operations
- Monitor token usage and costs

## 🔒 Security Guidelines

### Input Validation
- Validate all user inputs
- Sanitize file uploads
- Check file types and sizes
- Prevent path traversal attacks

### API Security
- Implement proper authentication
- Use HTTPS in production
- Add CORS protection
- Rate limit API endpoints

### Data Protection
- Never log sensitive data
- Use environment variables for secrets
- Implement proper error handling
- Follow OWASP guidelines

## 📋 Pull Request Checklist

Before submitting your PR, ensure:

- [ ] Code follows project style guidelines
- [ ] All tests pass (`pytest`)
- [ ] Code coverage meets requirements
- [ ] Documentation is updated
- [ ] Pre-commit hooks pass
- [ ] No sensitive data in commits
- [ ] PR description explains changes
- [ ] Related issues are referenced

## 🎯 Feature Request Process

### 1. Check Existing Issues
Search for similar feature requests first.

### 2. Create Detailed Issue
Include:
- Clear description of the feature
- Use cases and benefits
- Proposed implementation approach
- Potential challenges or considerations

### 3. Discussion
Engage with maintainers and community for feedback.

### 4. Implementation
Once approved, follow the development workflow above.

## 🐛 Bug Report Guidelines

### Include These Details
- **Environment**: OS, Python version, dependencies
- **Steps to Reproduce**: Clear, numbered steps
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Error Messages**: Full error logs
- **Screenshots**: If applicable

### Example Bug Report
```markdown
## Bug Description
Document analysis fails for files larger than 10MB

## Environment
- OS: Windows 11
- Python: 3.10.8
- Browser: Chrome 120

## Steps to Reproduce
1. Navigate to Analysis tab
2. Upload PDF file > 10MB
3. Click "Analyze"
4. Error occurs

## Expected Behavior
Large files should be processed successfully

## Actual Behavior
Gets "Request timeout" error after 30 seconds

## Error Message
```
TimeoutError: Request timed out after 30 seconds
```

## 🤝 Community Guidelines

### Be Respectful
- Use inclusive language
- Be patient with new contributors
- Provide constructive feedback
- Help others learn and grow

### Communication Channels
- **Issues**: Bug reports and feature requests
- **Discussions**: General questions and ideas
- **Pull Requests**: Code contributions
- **Email**: security@documentportal.com (security issues only)

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- Special thanks for major features

## 📞 Getting Help

### Stuck on Something?
1. Check existing documentation
2. Search closed issues
3. Ask in GitHub Discussions
4. Tag maintainers in issues (sparingly)

### Mentorship
New contributors can request mentorship for:
- First-time contributions
- Complex feature development
- Architecture decisions
- Best practices guidance

## 📈 Roadmap Priorities

Current focus areas:
1. **Performance Optimization** - Caching and response times
2. **Additional Document Formats** - More file type support
3. **Advanced Analytics** - Better metrics and monitoring
4. **Mobile Experience** - Responsive design improvements
5. **API Enhancements** - More powerful endpoints

Thank you for contributing to Enhanced Document Portal! 🚀
