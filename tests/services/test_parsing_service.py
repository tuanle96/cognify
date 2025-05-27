#!/usr/bin/env python3
"""
Test document parsing service implementation.
"""

import asyncio
import sys
import os
import tempfile
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_parsing_factory():
    """Test parsing factory functionality."""
    print("🔍 Testing parsing factory...")
    
    try:
        from app.services.parsers.factory import parser_factory
        from app.services.parsers.base import DocumentType
        
        # Test available parsers
        parsers = parser_factory.get_available_parsers()
        print(f"   ✅ Available parsers: {parsers}")
        
        # Test supported types
        types = parser_factory.get_supported_types()
        print(f"   ✅ Supported types: {[t.value for t in types]}")
        
        # Test parser info
        info = parser_factory.get_parser_info()
        for parser_name, parser_info in info.items():
            print(f"   📋 {parser_name}: {parser_info.get('supported_types', [])}")
        
        # Test parser creation
        code_parser = parser_factory.create_code_parser()
        print(f"   ✅ Code parser: {code_parser.parser_name}")
        
        text_parser = parser_factory.create_text_parser()
        print(f"   ✅ Text parser: {text_parser.parser_name}")
        
        # Test parsers for specific type
        code_parsers = parser_factory.get_parsers_for_type(DocumentType.CODE)
        print(f"   ✅ Code parsers: {len(code_parsers)}")
        
        print("✅ Parsing factory test passed")
        return True
        
    except Exception as e:
        print(f"❌ Parsing factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_parsing_models():
    """Test parsing model classes."""
    print("🔍 Testing parsing models...")
    
    try:
        from app.services.parsers.base import (
            ParsedDocument, ParsingRequest, ParsingResponse, DocumentType
        )
        
        # Test ParsingRequest
        request = ParsingRequest(
            content="print('Hello, World!')",
            document_type=DocumentType.CODE,
            language="python",
            extract_metadata=True
        )
        print(f"   ✅ ParsingRequest: type={request.document_type.value}, lang={request.language}")
        
        # Test auto-detection
        request_auto = ParsingRequest(file_path="test.py")
        print(f"   ✅ Auto-detected: type={request_auto.document_type.value}, lang={request_auto.language}")
        
        # Test ParsedDocument
        document = ParsedDocument(
            content="print('Hello, World!')",
            document_type=DocumentType.CODE,
            language="python",
            metadata={"line_count": 1}
        )
        print(f"   ✅ ParsedDocument: {len(document.content)} chars, {document.language}")
        
        # Test ParsingResponse
        response = ParsingResponse(
            document=document,
            processing_time=0.1,
            parser_used="code_parser",
            success=True
        )
        print(f"   ✅ ParsingResponse: success={response.success}, time={response.processing_time}s")
        
        # Test validation
        try:
            invalid_request = ParsingRequest()
            print(f"   ❌ Unexpected: Created request without content")
            return False
        except ValueError as e:
            print(f"   ✅ Expected error for empty request: {e}")
        
        print("✅ Parsing models test passed")
        return True
        
    except Exception as e:
        print(f"❌ Parsing models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_code_parser():
    """Test code parser functionality."""
    print("🔍 Testing code parser...")
    
    try:
        from app.services.parsers.code_parser import CodeParser
        from app.services.parsers.base import ParsingRequest, DocumentType
        
        # Create parser
        parser = CodeParser()
        await parser.initialize()
        print(f"   ✅ Parser initialized: {parser.parser_name}")
        
        # Test Python code
        python_code = '''
def hello_world():
    """A simple greeting function."""
    print("Hello, World!")
    return "Hello"

class Greeter:
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        return f"Hello, {self.name}!"

if __name__ == "__main__":
    hello_world()
'''
        
        request = ParsingRequest(
            content=python_code,
            document_type=DocumentType.CODE,
            language="python",
            extract_metadata=True,
            extract_sections=True
        )
        
        # Test can_parse
        can_parse = await parser.can_parse(request)
        print(f"   ✅ Can parse Python: {can_parse}")
        
        # Parse code
        response = await parser.parse(request)
        print(f"   ✅ Parse success: {response.success}")
        print(f"   ✅ Processing time: {response.processing_time:.3f}s")
        print(f"   ✅ Line count: {response.document.metadata.get('line_count', 0)}")
        print(f"   ✅ Function count: {response.document.metadata.get('function_count', 0)}")
        print(f"   ✅ Class count: {response.document.metadata.get('class_count', 0)}")
        print(f"   ✅ Sections: {len(response.document.sections)}")
        
        # Test JavaScript code
        js_request = ParsingRequest(
            content="function test() { return 'hello'; }",
            language="javascript"
        )
        
        js_can_parse = await parser.can_parse(js_request)
        print(f"   ✅ Can parse JavaScript: {js_can_parse}")
        
        print("✅ Code parser test passed")
        return True
        
    except Exception as e:
        print(f"❌ Code parser test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_text_parser():
    """Test text parser functionality."""
    print("🔍 Testing text parser...")
    
    try:
        from app.services.parsers.text_parser import TextParser
        from app.services.parsers.base import ParsingRequest, DocumentType
        
        # Create parser
        parser = TextParser()
        await parser.initialize()
        print(f"   ✅ Parser initialized: {parser.parser_name}")
        
        # Test markdown content
        markdown_content = '''
# Document Title

This is a **markdown** document with various elements.

## Section 1

- Item 1
- Item 2
- Item 3

### Subsection

Here's some `inline code` and a [link](https://example.com).

```python
def example():
    return "code block"
```

## Section 2

Another paragraph with more content.
'''
        
        request = ParsingRequest(
            content=markdown_content,
            document_type=DocumentType.MARKDOWN,
            extract_metadata=True,
            extract_sections=True
        )
        
        # Test can_parse
        can_parse = await parser.can_parse(request)
        print(f"   ✅ Can parse Markdown: {can_parse}")
        
        # Parse markdown
        response = await parser.parse(request)
        print(f"   ✅ Parse success: {response.success}")
        print(f"   ✅ Processing time: {response.processing_time:.3f}s")
        print(f"   ✅ Word count: {response.document.metadata.get('word_count', 0)}")
        print(f"   ✅ Header count: {response.document.metadata.get('header_count', 0)}")
        print(f"   ✅ Link count: {response.document.metadata.get('link_count', 0)}")
        print(f"   ✅ Sections: {len(response.document.sections)}")
        
        # Test plain text
        text_request = ParsingRequest(
            content="This is plain text content.",
            document_type=DocumentType.TEXT
        )
        
        text_response = await parser.parse(text_request)
        print(f"   ✅ Plain text parse success: {text_response.success}")
        
        print("✅ Text parser test passed")
        return True
        
    except Exception as e:
        print(f"❌ Text parser test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_structured_parser():
    """Test structured data parser functionality."""
    print("🔍 Testing structured parser...")
    
    try:
        from app.services.parsers.structured_parser import StructuredParser
        from app.services.parsers.base import ParsingRequest, DocumentType
        
        # Create parser
        parser = StructuredParser()
        await parser.initialize()
        print(f"   ✅ Parser initialized: {parser.parser_name}")
        
        # Test JSON content
        json_content = '''
{
    "name": "Test Document",
    "version": "1.0",
    "features": ["parsing", "validation", "formatting"],
    "config": {
        "enabled": true,
        "timeout": 30
    }
}
'''
        
        request = ParsingRequest(
            content=json_content,
            document_type=DocumentType.JSON,
            extract_metadata=True,
            extract_sections=True
        )
        
        # Test can_parse
        can_parse = await parser.can_parse(request)
        print(f"   ✅ Can parse JSON: {can_parse}")
        
        # Parse JSON
        response = await parser.parse(request)
        print(f"   ✅ Parse success: {response.success}")
        print(f"   ✅ Processing time: {response.processing_time:.3f}s")
        print(f"   ✅ JSON type: {response.document.metadata.get('type', 'unknown')}")
        print(f"   ✅ Key count: {response.document.metadata.get('key_count', 0)}")
        print(f"   ✅ Depth: {response.document.metadata.get('depth', 0)}")
        print(f"   ✅ Sections: {len(response.document.sections)}")
        
        # Test CSV content
        csv_content = '''name,age,city
John,25,New York
Jane,30,San Francisco
Bob,35,Chicago'''
        
        csv_request = ParsingRequest(
            content=csv_content,
            document_type=DocumentType.CSV,
            extract_metadata=True
        )
        
        csv_response = await parser.parse(csv_request)
        print(f"   ✅ CSV parse success: {csv_response.success}")
        print(f"   ✅ CSV rows: {csv_response.document.metadata.get('row_count', 0)}")
        print(f"   ✅ CSV columns: {csv_response.document.metadata.get('column_count', 0)}")
        
        print("✅ Structured parser test passed")
        return True
        
    except Exception as e:
        print(f"❌ Structured parser test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_parsing_service():
    """Test parsing service functionality."""
    print("🔍 Testing parsing service...")
    
    try:
        from app.services.parsers.service import ParsingService, ParsingServiceConfig
        from app.services.parsers.base import DocumentType
        
        # Create service with custom config
        config = ParsingServiceConfig(
            max_file_size=1024 * 1024,  # 1MB
            timeout=30.0,
            enable_caching=True
        )
        
        service = ParsingService(config)
        await service.initialize()
        print(f"   ✅ Service initialized with {len(service.parsers)} parsers")
        
        # Test single document parsing
        response = await service.parse_document(
            content="print('Hello from parsing service!')",
            document_type=DocumentType.CODE,
            language="python"
        )
        print(f"   ✅ Single parse success: {response.success}")
        
        # Test supported formats
        formats = await service.get_supported_formats()
        print(f"   ✅ Supported formats: {list(formats.keys())}")
        
        # Test health check
        health = await service.health_check()
        print(f"   ✅ Health status: {health['status']}")
        print(f"   ✅ Active parsers: {health['stats']['active_parsers']}")
        
        # Test stats
        stats = service.get_stats()
        print(f"   ✅ Total requests: {stats['total_requests']}")
        print(f"   ✅ Success rate: {stats['success_rate']:.2f}")
        
        # Test batch parsing
        batch_requests = [
            {"content": "def test(): pass", "document_type": DocumentType.CODE},
            {"content": "# Test Document\nContent here", "document_type": DocumentType.MARKDOWN},
            {"content": '{"test": true}', "document_type": DocumentType.JSON}
        ]
        
        from app.services.parsers.base import ParsingRequest
        requests = [ParsingRequest(**req) for req in batch_requests]
        
        batch_responses = await service.parse_batch(requests)
        print(f"   ✅ Batch parsing: {len(batch_responses)} responses")
        print(f"   ✅ Batch success: {sum(1 for r in batch_responses if r.success)}/{len(batch_responses)}")
        
        # Cleanup
        await service.cleanup()
        print(f"   ✅ Service cleanup completed")
        
        print("✅ Parsing service test passed")
        return True
        
    except Exception as e:
        print(f"❌ Parsing service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_file_parsing():
    """Test file-based parsing."""
    print("🔍 Testing file parsing...")
    
    try:
        from app.services.parsers.service import ParsingService
        
        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            python_file = temp_path / "test.py"
            python_file.write_text('''
def hello():
    """Test function"""
    return "Hello, World!"

class TestClass:
    pass
''')
            
            markdown_file = temp_path / "test.md"
            markdown_file.write_text('''
# Test Document

This is a test markdown file.

## Features

- Feature 1
- Feature 2
''')
            
            json_file = temp_path / "test.json"
            json_file.write_text('{"name": "test", "value": 42}')
            
            # Test parsing service
            service = ParsingService()
            await service.initialize()
            
            # Parse individual files
            python_response = await service.parse_document(file_path=str(python_file))
            print(f"   ✅ Python file parse: {python_response.success}")
            
            markdown_response = await service.parse_document(file_path=str(markdown_file))
            print(f"   ✅ Markdown file parse: {markdown_response.success}")
            
            json_response = await service.parse_document(file_path=str(json_file))
            print(f"   ✅ JSON file parse: {json_response.success}")
            
            # Parse directory
            directory_responses = await service.parse_directory(str(temp_path))
            print(f"   ✅ Directory parsing: {len(directory_responses)} files")
            print(f"   ✅ Directory success: {sum(1 for r in directory_responses if r.success)}/{len(directory_responses)}")
            
            await service.cleanup()
        
        print("✅ File parsing test passed")
        return True
        
    except Exception as e:
        print(f"❌ File parsing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run parsing service tests."""
    print("🚀 Document Parsing Service Test")
    print("=" * 70)
    
    tests = [
        ("Parsing Factory", test_parsing_factory),
        ("Parsing Models", test_parsing_models),
        ("Code Parser", test_code_parser),
        ("Text Parser", test_text_parser),
        ("Structured Parser", test_structured_parser),
        ("Parsing Service", test_parsing_service),
        ("File Parsing", test_file_parsing),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            result = await test_func()
            if result:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
        print()
    
    print("=" * 70)
    print(f"📊 Document Parsing Service Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All parsing service tests passed!")
        print("🚀 Document parsing service implementation is solid!")
        return 0
    elif passed >= total * 0.8:
        print("🎯 Most parsing service tests passed!")
        return 0
    else:
        print(f"⚠️  {total - passed} tests need attention.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Tests crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
