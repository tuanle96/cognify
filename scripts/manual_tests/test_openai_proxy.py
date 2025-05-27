#!/usr/bin/env python3
"""
Test OpenAI proxy directly to debug connection issues.
"""

import asyncio
import aiohttp
import json

async def test_openai_proxy():
    """Test OpenAI proxy connection directly."""
    import os

    # Get configuration from environment
    base_url = os.getenv("OPENAI_BASE_URL", "https://ai.earnbase.io/v1")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set it in your .env file or environment:")
        print("export OPENAI_API_KEY=your-api-key-here")
        return False

    print("🔍 Testing OpenAI proxy connection...")
    print(f"   Base URL: {base_url}")
    print(f"   API Key: {api_key[:20]}...")

    # Test 1: Basic health check
    print("\n📋 Test 1: Basic HTTP connection")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url) as response:
                print(f"   Status: {response.status}")
                text = await response.text()
                print(f"   Response: {text[:200]}...")
    except Exception as e:
        print(f"   ❌ Basic connection failed: {e}")

    # Test 2: Chat completions endpoint
    print("\n📋 Test 2: Chat completions endpoint")
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "Hello! Please respond with exactly: 'Proxy working correctly'"}
            ],
            "max_tokens": 50,
            "temperature": 0
        }

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers
            ) as response:
                print(f"   Status: {response.status}")

                if response.status == 200:
                    data = await response.json()
                    content = data["choices"][0]["message"]["content"]
                    print(f"   ✅ Success! Response: {content}")
                    return True
                else:
                    error_text = await response.text()
                    print(f"   ❌ Error: {error_text}")
                    return False

    except Exception as e:
        print(f"   ❌ Chat completions failed: {e}")
        return False

    # Test 3: Embeddings endpoint
    print("\n📋 Test 3: Embeddings endpoint")
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "text-embedding-004",
            "input": ["test embedding"]
        }

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.post(
                f"{base_url}/embeddings",
                json=payload,
                headers=headers
            ) as response:
                print(f"   Status: {response.status}")

                if response.status == 200:
                    data = await response.json()
                    embedding_length = len(data["data"][0]["embedding"])
                    print(f"   ✅ Success! Embedding dimensions: {embedding_length}")
                    return True
                else:
                    error_text = await response.text()
                    print(f"   ❌ Error: {error_text}")
                    return False

    except Exception as e:
        print(f"   ❌ Embeddings failed: {e}")
        return False

async def main():
    """Run proxy tests."""
    print("🚀 OpenAI Proxy Connection Test")
    print("=" * 50)

    success = await test_openai_proxy()

    print("\n" + "=" * 50)
    if success:
        print("🎉 OpenAI proxy is working correctly!")
        print("   You can now use Cognify with real AI services.")
    else:
        print("❌ OpenAI proxy connection failed.")
        print("   Please check your API key and network connection.")

if __name__ == "__main__":
    asyncio.run(main())
