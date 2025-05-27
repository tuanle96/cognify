#!/usr/bin/env python3
"""
Test RAG Search Functionality
Tests document upload and search capabilities on Cognify RAG system.
Moved from root to tests/manual/ for better organization
"""

import requests
import json
import time
from pathlib import Path

def load_test_content():
    """Load test content for RAG testing."""
    # Try to load from examples directory first
    test_files = [
        "examples/comprehensive_test_content.py",
        "examples/snake_game.py", 
        "snake_game.py"  # fallback to root if still there
    ]
    
    for file_path in test_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content:
                    print(f"✅ Loaded test content from: {file_path}")
                    return content
        except:
            continue
    
    # If no file found, create sample content
    return """
def sample_function():
    '''Sample Python function for testing RAG search.'''
    return "This is sample content for testing document search capabilities."

class SampleClass:
    '''Sample class for testing code analysis.'''
    def __init__(self):
        self.data = []
    
    def add_item(self, item):
        '''Add item to the collection.'''
        self.data.append(item)
        return len(self.data)
"""

def test_rag_search():
    """Test complete RAG workflow: auth -> collection -> document -> search."""
    base_url = "http://localhost:8002"
    
    # Test user
    user_data = {
        "email": "rag_tester@example.com",
        "password": "RagTest123!",
        "full_name": "RAG Tester"
    }
    
    session = requests.Session()
    session.headers.update({'Content-Type': 'application/json'})
    
    print("🔐 Testing Authentication...")
    
    # Register user
    try:
        response = session.post(f"{base_url}/api/v1/auth/register", json=user_data)
        if response.status_code in [200, 201]:
            print("✅ User registered successfully")
        else:
            print(f"⚠️ Registration: {response.status_code}")
    except Exception as e:
        print(f"❌ Registration error: {e}")
    
    # Login
    try:
        login_data = {"email": user_data["email"], "password": user_data["password"]}
        response = session.post(f"{base_url}/api/v1/auth/login", json=login_data)
        if response.status_code == 200:
            data = response.json()
            token = data.get('access_token')
            session.headers.update({'Authorization': f'Bearer {token}'})
            print("✅ Login successful, token obtained")
        else:
            print(f"❌ Login failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Login error: {e}")
        return
    
    print("\n📁 Creating Collection...")
    
    # Create collection
    collection_data = {
        "name": "RAG Test Collection",
        "description": "Collection for testing RAG search functionality"
    }
    
    try:
        response = session.post(f"{base_url}/api/v1/collections/", json=collection_data)
        if response.status_code in [200, 201]:
            collection = response.json()
            collection_id = collection.get('id')
            print(f"✅ Collection created: {collection_id}")
        else:
            print(f"❌ Collection creation failed: {response.status_code} - {response.text}")
            return
    except Exception as e:
        print(f"❌ Collection error: {e}")
        return
    
    print("\n📄 Uploading Document...")
    
    # Load and upload test content
    test_content = load_test_content()
    if not test_content:
        print("❌ Could not load test content")
        return
    
    document_data = {
        "title": "Test Code Implementation",
        "content": test_content,
        "content_type": "python_code",
        "collection_id": collection_id,
        "metadata": {
            "filename": "test_content.py",
            "language": "python",
            "lines_of_code": len(test_content.split('\n')),
            "description": "Test content for RAG search functionality"
        }
    }
    
    try:
        response = session.post(f"{base_url}/api/v1/documents/", json=document_data)
        if response.status_code in [200, 201]:
            document = response.json()
            document_id = document.get('id')
            print(f"✅ Document uploaded: {document_id}")
            print(f"   Content size: {len(test_content)} characters")
        else:
            print(f"❌ Document upload failed: {response.status_code} - {response.text}")
            return
    except Exception as e:
        print(f"❌ Document upload error: {e}")
        return
    
    # Wait for indexing
    print("\n⏳ Waiting for indexing...")
    time.sleep(3)
    
    print("\n🔍 Testing RAG Search Queries...")
    
    # Test queries
    test_queries = [
        "How does the function work?",
        "What classes are defined?", 
        "How is data stored?",
        "What methods are available?",
        "How to add items?",
        "What is the sample function?",
        "How is the class implemented?",
        "What are the main features?",
        "How does the code work?",
        "What functionality is provided?"
    ]
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        
        query_data = {
            "query": query,
            "collection_id": collection_id,
            "top_k": 3,
            "include_metadata": True
        }
        
        try:
            response = session.post(f"{base_url}/api/v1/query/search", json=query_data)
            if response.status_code == 200:
                data = response.json()
                search_results = data.get('results', [])
                
                print(f"✅ Found {len(search_results)} results")
                
                for j, result in enumerate(search_results, 1):
                    score = result.get('score', 0)
                    content = result.get('content', '')[:200] + "..."
                    print(f"   {j}. Score: {score:.3f}")
                    print(f"      Content: {content}")
                
                results.append({
                    'query': query,
                    'results_count': len(search_results),
                    'top_score': search_results[0].get('score', 0) if search_results else 0,
                    'success': True
                })
                
            else:
                print(f"❌ Query failed: {response.status_code} - {response.text}")
                results.append({
                    'query': query,
                    'success': False,
                    'error': response.text
                })
                
        except Exception as e:
            print(f"❌ Query error: {e}")
            results.append({
                'query': query,
                'success': False,
                'error': str(e)
            })
        
        time.sleep(0.5)  # Small delay between queries
    
    # Summary
    print("\n" + "="*60)
    print("RAG SEARCH TEST SUMMARY")
    print("="*60)
    
    successful_queries = [r for r in results if r.get('success')]
    failed_queries = [r for r in results if not r.get('success')]
    
    print(f"✅ Successful queries: {len(successful_queries)}/{len(results)}")
    print(f"❌ Failed queries: {len(failed_queries)}")
    
    if successful_queries:
        avg_score = sum(r.get('top_score', 0) for r in successful_queries) / len(successful_queries)
        avg_results = sum(r.get('results_count', 0) for r in successful_queries) / len(successful_queries)
        
        print(f"📊 Average top score: {avg_score:.3f}")
        print(f"📊 Average results per query: {avg_results:.1f}")
        
        print(f"\n🏆 Best performing queries:")
        best_queries = sorted(successful_queries, key=lambda x: x.get('top_score', 0), reverse=True)[:3]
        for i, q in enumerate(best_queries, 1):
            print(f"   {i}. \"{q['query']}\" (score: {q['top_score']:.3f})")
    
    if failed_queries:
        print(f"\n❌ Failed queries:")
        for q in failed_queries:
            print(f"   - \"{q['query']}\": {q.get('error', 'Unknown error')}")
    
    # Save results
    timestamp = int(time.time())
    filename = f"tests/manual/rag_search_results_{timestamp}.json"
    
    # Ensure directory exists
    Path("tests/manual").mkdir(parents=True, exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump({
            'test_summary': {
                'total_queries': len(results),
                'successful': len(successful_queries),
                'failed': len(failed_queries),
                'success_rate': len(successful_queries) / len(results) * 100 if results else 0,
                'avg_top_score': avg_score if successful_queries else 0,
                'avg_results_count': avg_results if successful_queries else 0
            },
            'detailed_results': results,
            'timestamp': timestamp
        }, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")
    print("="*60)

if __name__ == "__main__":
    test_rag_search()
