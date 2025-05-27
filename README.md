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
- OpenAI API key or compatible proxy
- Git

### Installation

1. **Clone and Install**
```bash
git clone https://github.com/tuanle96/cognify.git
cd cognify
poetry install
```

2. **Configure Environment**
```bash
# Copy template and customize
cp .env.example .env.development
nano .env.development  # Add your API keys
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

### 🐳 Docker Deployment

```bash
# Development with database
docker-compose up -d

# Production deployment
docker-compose -f docker-compose.production.yml up -d

# Database only
docker-compose -f docker-compose.database.yml up -d
```

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
.env.example      # Complete template
.env.development  # Development config
.env.production   # Production config

# Setup for development
cp .env.example .env.development
nano .env.development  # Add your API keys

# Key settings
LITELLM_API_KEY=your-api-key
DEFAULT_LLM_PROVIDER=openai
CHUNKING_STRATEGY=hybrid
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
