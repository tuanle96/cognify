# Cognify - AI-Powered Codebase Analysis

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![AI Services](https://img.shields.io/badge/AI_Services-Operational-success.svg)](cognify/)

**Cognify** is an intelligent RAG system for codebase analysis featuring AI Agent-driven chunking technology that delivers superior semantic understanding and context preservation.

## ✨ Key Features

- 🧠 **Agentic Chunking**: AI agents make intelligent chunking decisions for optimal semantic coherence
- 🎯 **Purpose-Driven Processing**: Adaptive strategies for code review, bug detection, and documentation
- 🔍 **Enhanced Code Search**: Semantic search with AI-optimized chunks
- 🤖 **Multi-Agent Coordination**: Specialized agents collaborate for analysis and processing
- ⚡ **Smart Performance**: Hybrid approach with intelligent caching and adaptive processing
- 🌐 **Multi-language Support**: Python, JavaScript, Go, Java with language-specific optimization

## 🛠️ Technology Stack

- **API**: FastAPI with async/await support
- **AI/ML**: OpenAI GPT-4, LangChain/CrewAI, Tree-sitter parsing
- **Database**: PostgreSQL (metadata), Qdrant (vectors), Redis (cache)
- **Infrastructure**: Docker, Kubernetes, Prometheus + Grafana

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (recommended)
- Git
- PostgreSQL, Redis, Qdrant (if running locally)

### Installation

1. **Clone and Install**
```bash
# Clone the repository
git clone <repository-url>
cd cognify
poetry install
```

2. **Configure Environment**
```bash
# Copy template and customize
cp .env.example .env
nano .env  # Update configuration for your environment
```

3. **Start Application**
```bash
# Start the API server
poetry run python -m app.main

# Or with uvicorn
poetry run uvicorn app.main:app --reload
```

4. **Verify Installation**
```bash
# Check API health
curl http://localhost:8000/health

# Access interactive docs
open http://localhost:8000/docs

# Test chunking API
curl -X POST "http://localhost:8000/api/v1/chunk" \
  -H "Content-Type: application/json" \
  -d '{"content": "def hello(): return \"world\"", "language": "python"}'
```

### 🐳 Docker Deployment (Recommended)

**Prerequisites**: Docker & Docker Compose installed

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with your settings (database passwords, etc.)

# 2. Start all services
docker-compose up -d

# 3. Check service status
docker-compose ps

# 4. View logs
docker-compose logs -f cognify-api

# 5. Access the application
# API: http://localhost:30000
# Docs: http://localhost:30000/docs
# Health: http://localhost:30000/health
# Nginx Proxy: http://localhost:30005
# Grafana: http://localhost:30008

# Stop all services
docker-compose down
```

**Services & Port Mapping (127.0.0.1 for security):**
- `cognify-api`: FastAPI application (30000 → 8000)
- `postgres`: PostgreSQL database (30001 → 5432)
- `redis`: Redis cache (30002 → 6379)
- `qdrant`: Vector database (30003 → 6333, 30004 → 6334)
- `nginx`: Reverse proxy (30005 → 80, 30006 → 443)
- `prometheus`: Monitoring (30007 → 9090)
- `grafana`: Dashboard (30008 → 3000)

## 📖 API Usage

### Basic Chunking
```python
import httpx

# Chunk code for review
response = httpx.post("http://localhost:8000/api/v1/chunk", json={
    "content": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "language": "python",
    "purpose": "code_review"
})

print(f"Generated {response.json()['chunk_count']} chunks")
```

### Purpose-Driven Processing
```python
# Different purposes optimize chunking strategy
purposes = ["code_review", "bug_detection", "documentation"]

for purpose in purposes:
    response = httpx.post("http://localhost:8000/api/v1/chunk", json={
        "content": source_code,
        "language": "python",
        "purpose": purpose
    })
    print(f"{purpose}: {response.json()['strategy_used']}")
```

## 🧪 Development

### Running Tests
```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app --cov=agents

# Run specific test types
poetry run pytest -m unit
poetry run pytest -m integration
```

### Code Quality
```bash
# Format code
poetry run black .
poetry run isort .

# Type checking
poetry run mypy .
```

## 🛠️ Configuration

### Environment Setup
```bash
# Available environment files
.env.example              # Development template
.env.production.example   # Production template

# Setup for development
cp .env.example .env
nano .env  # Update configuration

# Key settings (all managed via database in production)
ENVIRONMENT=development
DEBUG=true
DATABASE_URL=postgresql://cognify:password@localhost:5432/cognify_dev
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379/0
```

### LLM Configuration
```bash
# LLM settings are now managed via database
# Use the admin API to configure providers:
curl -X POST "http://localhost:8000/api/v1/admin/llm-providers" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "openai",
    "provider_type": "openai",
    "api_key": "your-api-key",
    "base_url": "https://api.openai.com/v1"
  }'
```

### Security Best Practices
- Never commit real secrets to version control
- Use `.env.example` as template only
- Change all placeholder values in production
- Restrict CORS origins in production

## 🗺️ Roadmap

### Current Focus
- Enhanced multi-language support (JavaScript, TypeScript, Go, Java, Rust, C++)
- Advanced agent coordination and performance optimization
- Enterprise features and security enhancements

### Upcoming Features
- Python SDK for programmatic access
- Real-time codebase synchronization
- IDE integrations (VS Code, JetBrains)
- CI/CD pipeline integration

## 📚 Documentation

- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [Architecture Guide](../docs/agentic_chunking_strategy.md) - Detailed architecture
- [Development Setup](../docs/development_setup.md) - Development environment
- [Deployment Guide](../docs/deployment_guide.md) - Production deployment

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- AI services powered by [LiteLLM](https://litellm.ai/) and [OpenAI](https://openai.com/)
- Vector storage with [Qdrant](https://qdrant.tech/)
- Code parsing with [Tree-sitter](https://tree-sitter.github.io/tree-sitter/)

---

**Cognify** - Where Code Meets Intelligence 🧠✨
