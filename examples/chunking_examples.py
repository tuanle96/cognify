#!/usr/bin/env python3
"""
Cognify Chunking Examples
Comprehensive examples of using Cognify's agentic chunking API.
"""

import httpx
import asyncio
from typing import Dict, Any


# =============================================================================
# BASIC CHUNKING EXAMPLES
# =============================================================================

def basic_chunking_example():
    """Basic code chunking example."""

    # Simple function chunking
    response = httpx.post("http://localhost:8001/api/v1/chunking/", json={
        "content": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "language": "python",
        "strategy": "hybrid"
    })

    result = response.json()
    print(f"Generated {result['total_chunks']} chunks")
    print(f"Processing time: {result['processing_time']:.4f}s")
    print(f"Strategy used: {result['strategy_used']}")


def purpose_driven_chunking():
    """Demonstrate different chunking strategies for different purposes."""

    sample_code = """
class OrderProcessor:
    def __init__(self, payment_service, inventory_service):
        self.payment_service = payment_service
        self.inventory_service = inventory_service

    def create_order(self, items, customer_id):
        try:
            # Validate items
            for item in items:
                if not self.inventory_service.check_availability(item):
                    raise ValueError(f"Item {item} not available")

            # Calculate total
            total = sum(item.price * item.quantity for item in items)

            # Process payment
            payment_result = self.payment_service.charge(customer_id, total)
            if not payment_result.success:
                raise PaymentError("Payment failed")

            # Create order
            order = Order(items=items, customer_id=customer_id, total=total)
            return order

        except Exception as e:
            self.handle_error(e)
            raise

    def handle_error(self, error):
        # Log error and cleanup
        logger.error(f"Order processing failed: {error}")
        # Cleanup logic here
    """

    purposes = ["code_review", "bug_detection", "documentation"]

    for purpose in purposes:
        print(f"\n=== {purpose.upper()} CHUNKING ===")

        response = httpx.post("http://localhost:8001/api/v1/chunking/", json={
            "content": sample_code,
            "language": "python",
            "strategy": "hybrid"
        })

        result = response.json()
        print(f"Strategy: {result['strategy_used']}")
        print(f"Chunks: {result['total_chunks']}")
        print(f"Processing time: {result['processing_time']:.4f}s")

        # Print first chunk as example
        if result['chunks']:
            print(f"First chunk preview: {result['chunks'][0]['content'][:100]}...")


# =============================================================================
# ADVANCED USAGE EXAMPLES
# =============================================================================

async def streaming_chunking_example():
    """Example of streaming chunking for large codebases."""

    large_codebase = """
    # Large codebase content here...
    # This would typically be read from multiple files
    """ * 100  # Simulate large content

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8001/api/v1/chunking/",
            json={
                "content": large_codebase,
                "language": "python",
                "strategy": "agentic"
            }
        ) as response:
            async for chunk in response.aiter_text():
                print(f"Processing: {chunk}", end="", flush=True)


def batch_chunking_example():
    """Example of processing multiple files in batch."""

    files_to_process = [
        {"path": "models.py", "content": "class User: pass", "language": "python"},
        {"path": "views.py", "content": "def index(): return 'Hello'", "language": "python"},
        {"path": "utils.js", "content": "function helper() { return true; }", "language": "javascript"},
    ]

    results = []

    for file_info in files_to_process:
        response = httpx.post("http://localhost:8001/api/v1/chunking/", json={
            "content": file_info["content"],
            "language": file_info["language"],
            "strategy": "hybrid"
        })

        result = response.json()
        results.append({
            "file": file_info["path"],
            "chunks": result["total_chunks"],
            "processing_time": result["processing_time"]
        })

    print("Batch processing results:")
    for result in results:
        print(f"  {result['file']}: {result['chunks']} chunks (time: {result['processing_time']:.4f}s)")


# =============================================================================
# COMPARISON EXAMPLES
# =============================================================================

def chunking_comparison_example():
    """Compare traditional vs agentic chunking."""

    ecommerce_code = """
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

class OrderStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@dataclass
class OrderItem:
    product_id: str
    quantity: int
    price: float

    def get_total(self) -> float:
        return self.quantity * self.price

    def validate(self) -> bool:
        return self.quantity > 0 and self.price >= 0

class Order:
    def __init__(self, customer_id: str, items: List[OrderItem]):
        self.customer_id = customer_id
        self.items = items
        self.status = OrderStatus.PENDING
        self.total = self.calculate_total()

    def calculate_total(self) -> float:
        return sum(item.get_total() for item in self.items)

    def add_item(self, item: OrderItem):
        self.items.append(item)
        self.total = self.calculate_total()

    def remove_item(self, product_id: str):
        self.items = [item for item in self.items if item.product_id != product_id]
        self.total = self.calculate_total()

class OrderProcessor:
    def __init__(self, payment_service, inventory_service):
        self.payment_service = payment_service
        self.inventory_service = inventory_service

    def create_order(self, customer_id: str, items: List[OrderItem]) -> Order:
        # Validate all items
        for item in items:
            if not item.validate():
                raise ValueError(f"Invalid item: {item}")
            if not self.inventory_service.check_availability(item.product_id, item.quantity):
                raise ValueError(f"Insufficient inventory for {item.product_id}")

        order = Order(customer_id, items)
        return order

    def process_payment(self, order: Order) -> bool:
        try:
            result = self.payment_service.charge(order.customer_id, order.total)
            if result.success:
                order.status = OrderStatus.PROCESSING
                self._reserve_inventory(order)
                return True
            return False
        except Exception as e:
            order.status = OrderStatus.CANCELLED
            raise PaymentError(f"Payment failed: {e}")

    def _reserve_inventory(self, order: Order):
        for item in order.items:
            self.inventory_service.reserve(item.product_id, item.quantity)
    """

    print("=== CHUNKING COMPARISON ===")
    print("\nTraditional RAG (Rule-based) would create:")
    print("Chunk 1: Lines 1-20   (Enums + partial OrderItem)")
    print("Chunk 2: Lines 21-40  (OrderItem methods + partial Order)")
    print("Chunk 3: Lines 41-60  (Order methods split)")
    print("Chunk 4: Lines 61-80  (OrderProcessor init + partial create_order)")
    print("Chunk 5: Lines 81-100 (Rest of create_order + partial process_payment)")
    print("Problems: Logic scattered, context lost, methods separated from classes")

    print("\nCognify (AI Agent-driven) creates:")
    response = httpx.post("http://localhost:8001/api/v1/chunking/", json={
        "content": ecommerce_code,
        "language": "python",
        "strategy": "hybrid"
    })

    result = response.json()
    for i, chunk in enumerate(result['chunks'], 1):
        chunk_type = chunk['metadata'].get('chunk_type', 'semantic_unit')
        print(f"Chunk {i}: {chunk_type} ({len(chunk['content'])} chars)")

    print(f"Benefits: Semantic coherence, complete business logic, better context")
    print(f"Processing time: {result['processing_time']:.4f}s")
    print(f"Total chunks: {result['total_chunks']}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🚀 Cognify Chunking Examples")
    print("=" * 50)

    try:
        # Basic examples
        print("\n1. Basic Chunking")
        basic_chunking_example()

        print("\n2. Purpose-Driven Chunking")
        purpose_driven_chunking()

        print("\n3. Batch Processing")
        batch_chunking_example()

        print("\n4. Chunking Comparison")
        chunking_comparison_example()

        # Advanced examples (async)
        print("\n5. Streaming Chunking")
        asyncio.run(streaming_chunking_example())

        print("\n✅ All examples completed successfully!")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Make sure Cognify API is running on http://localhost:8001")
