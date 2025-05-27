# Cognify Examples

This directory contains comprehensive examples of using Cognify's agentic chunking API.

## 📁 Available Examples

### `chunking_examples.py`
Comprehensive Python examples demonstrating:

- **Basic Chunking**: Simple code chunking with different purposes
- **Purpose-Driven Processing**: How different purposes affect chunking strategy
- **Batch Processing**: Processing multiple files efficiently
- **Streaming Chunking**: Handling large codebases with streaming
- **Comparison Examples**: Traditional vs agentic chunking comparison

## 🚀 Running Examples

### Prerequisites
- Cognify API running on `http://localhost:8000`
- Python with `httpx` installed

### Run Examples
```bash
# Install dependencies
pip install httpx

# Run all examples
python examples/chunking_examples.py

# Or run specific sections by modifying the main function
```

## 📖 Example Categories

### 1. Basic Usage
```python
# Simple chunking
response = httpx.post("http://localhost:8000/api/v1/chunk", json={
    "content": "def hello(): return 'world'",
    "language": "python",
    "purpose": "code_review"
})
```

### 2. Purpose-Driven Chunking
- `code_review`: Optimized for human review
- `bug_detection`: Focused on error-prone patterns
- `documentation`: Grouped by public APIs

### 3. Advanced Features
- Streaming for large codebases
- Batch processing multiple files
- Quality metrics and strategy comparison

## 🎯 Use Cases Demonstrated

1. **Code Review Workflow**: Chunks optimized for reviewable units
2. **Bug Detection**: Focus on error handling and edge cases
3. **Documentation Generation**: Public APIs and interfaces
4. **Semantic Search**: Context-aware code discovery

## 📚 Additional Resources

- [API Documentation](http://localhost:8000/docs)
- [Main README](../README.md)
- [Architecture Guide](../../docs/agentic_chunking_strategy.md)
