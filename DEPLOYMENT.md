# Deployment Guide - Document Portal

This guide covers deployment options for the enhanced Document Portal application.

## 🚀 Quick Deployment

### Local Development
```bash
# 1. Clone and setup
git clone https://github.com/sunnysavita10/document_portal.git
cd document_portal

# 2. Environment setup
conda create -p venv python=3.10 -y
conda activate ./venv
pip install -r requirements.txt

# 3. Configure environment
cp env.example .env
# Edit .env with your API keys

# 4. Start application
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

### Docker Deployment
```bash
# Build image
docker build -t document-portal:latest .

# Run with environment file
docker run -d \
  --name document-portal \
  -p 8080:8080 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/cache:/app/cache \
  document-portal:latest
```

## 🏗️ Production Deployment

### Prerequisites
- Python 3.10+
- Redis (recommended for caching)
- PostgreSQL/MongoDB (optional for advanced features)
- Nginx (for reverse proxy)
- SSL certificates

### Infrastructure Setup

#### 1. Redis Setup
```bash
# Install Redis
sudo apt update
sudo apt install redis-server

# Configure Redis
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Test connection
redis-cli ping
```

#### 2. Database Setup (Optional)
```bash
# PostgreSQL
sudo apt install postgresql postgresql-contrib
sudo -u postgres createdb document_portal

# MongoDB
sudo apt install mongodb
sudo systemctl enable mongodb
sudo systemctl start mongodb
```

#### 3. Application Deployment
```bash
# Create application user
sudo useradd -m -s /bin/bash docportal
sudo su - docportal

# Clone repository
git clone https://github.com/sunnysavita10/document_portal.git
cd document_portal

# Setup Python environment
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp env.example .env
# Edit .env with production settings
```

#### 4. Systemd Service
Create `/etc/systemd/system/document-portal.service`:
```ini
[Unit]
Description=Document Portal API
After=network.target

[Service]
Type=simple
User=docportal
WorkingDirectory=/home/docportal/document_portal
Environment=PATH=/home/docportal/document_portal/venv/bin
ExecStart=/home/docportal/document_portal/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8080
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable document-portal
sudo systemctl start document-portal
sudo systemctl status document-portal
```

#### 5. Nginx Configuration
Create `/etc/nginx/sites-available/document-portal`:
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL Configuration
    ssl_certificate /path/to/your/certificate.crt;
    ssl_certificate_key /path/to/your/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    
    # File upload limits
    client_max_body_size 100M;
    
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Static files
    location /static/ {
        alias /home/docportal/document_portal/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/document-portal /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## ☁️ Cloud Deployment

### AWS Deployment

#### Using EC2
```bash
# Launch EC2 instance (Ubuntu 22.04 LTS)
# Security group: Allow HTTP (80), HTTPS (443), SSH (22)

# Connect and setup
ssh -i your-key.pem ubuntu@your-ec2-ip

# Follow production deployment steps above
```

#### Using ECS with Fargate
```yaml
# docker-compose.yml for ECS
version: '3.8'
services:
  document-portal:
    image: your-account.dkr.ecr.region.amazonaws.com/document-portal:latest
    ports:
      - "8080:8080"
    environment:
      - REDIS_URL=redis://your-elasticache-endpoint:6379
      - DATABASE_URL=postgresql://user:pass@your-rds-endpoint:5432/dbname
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '1'
          memory: 2G
```

### Google Cloud Platform

#### Using Cloud Run
```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT-ID/document-portal

# Deploy to Cloud Run
gcloud run deploy document-portal \
  --image gcr.io/PROJECT-ID/document-portal \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --set-env-vars REDIS_URL=redis://your-memorystore-ip:6379
```

### Azure Deployment

#### Using Container Instances
```bash
# Create resource group
az group create --name document-portal-rg --location eastus

# Deploy container
az container create \
  --resource-group document-portal-rg \
  --name document-portal \
  --image your-registry/document-portal:latest \
  --dns-name-label document-portal-unique \
  --ports 8080 \
  --environment-variables \
    REDIS_URL=redis://your-redis-cache.redis.cache.windows.net:6380 \
  --secure-environment-variables \
    OPENAI_API_KEY=your-api-key
```

## 🔧 Configuration Management

### Environment Variables
```bash
# Production .env
DEBUG=false
LOG_LEVEL=INFO
REDIS_URL=redis://production-redis:6379
DATABASE_URL=postgresql://user:pass@prod-db:5432/docportal
CORS_ORIGINS=https://your-domain.com
RATE_LIMIT_PER_MINUTE=100
SESSION_CLEANUP_DAYS=7
```

### Secrets Management
```bash
# Using AWS Secrets Manager
aws secretsmanager create-secret \
  --name document-portal/api-keys \
  --secret-string '{"OPENAI_API_KEY":"sk-...","GROQ_API_KEY":"gsk_..."}'

# Using Azure Key Vault
az keyvault secret set \
  --vault-name your-keyvault \
  --name openai-api-key \
  --value "sk-your-key"
```

## 📊 Monitoring & Logging

### Application Monitoring
```python
# Add to main.py
import logging
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@app.middleware("http")
async def add_metrics(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_DURATION.observe(duration)
    
    return response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Log Configuration
```python
# logging_config.py
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "json": {
            "format": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "json",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"],
    },
}
```

## 🔒 Security Considerations

### API Security
- Use HTTPS in production
- Implement rate limiting
- Validate all inputs
- Use API keys securely
- Enable CORS properly

### Data Security
- Encrypt sensitive data at rest
- Use secure database connections
- Implement proper access controls
- Regular security updates

### Infrastructure Security
- Use firewalls and security groups
- Regular system updates
- Monitor for vulnerabilities
- Backup strategies

## 🚨 Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Monitor memory
htop
docker stats

# Solutions
- Increase memory limits
- Optimize caching strategy
- Implement pagination
```

#### Slow Response Times
```bash
# Check Redis connection
redis-cli ping

# Monitor database
# Check API rate limits
# Review caching strategy
```

#### API Key Issues
```bash
# Verify environment variables
env | grep API_KEY

# Check API key validity
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models
```

### Health Checks
```bash
# Application health
curl http://localhost:8080/health

# Detailed health with metrics
curl http://localhost:8080/health/detailed

# Cache statistics
curl http://localhost:8080/analytics/cache-stats
```

## 📈 Scaling Considerations

### Horizontal Scaling
- Use load balancers
- Implement session affinity
- Share cache across instances
- Database connection pooling

### Performance Optimization
- Enable Redis clustering
- Use CDN for static files
- Implement response compression
- Optimize database queries

### Cost Optimization
- Monitor token usage
- Implement caching strategies
- Use spot instances where appropriate
- Regular cleanup of old data
