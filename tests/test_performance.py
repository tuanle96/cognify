"""
Comprehensive performance and load tests.

Tests response times, throughput, memory usage, and scalability
characteristics of all API endpoints using mock implementations.
"""

import pytest
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import status


class TestResponseTimes:
    """Test API response time performance."""

    def test_auth_endpoints_response_time(self, mock_client, performance_timer):
        """Test authentication endpoints response times."""
        endpoints = [
            ("POST", "/api/v1/auth/register", {"email": "test@example.com", "password": "Test123"}),
            ("POST", "/api/v1/auth/login", {"email": "test@example.com", "password": "Test123"}),
            ("GET", "/api/v1/auth/me", None),
            ("POST", "/api/v1/auth/logout", None),
            ("POST", "/api/v1/auth/refresh", None)
        ]
        
        for method, endpoint, data in endpoints:
            performance_timer.start()
            
            if method == "GET":
                response = mock_client.get(endpoint)
            else:
                response = mock_client.post(endpoint, json=data)
            
            performance_timer.stop()
            
            assert response.status_code == status.HTTP_200_OK
            assert performance_timer.elapsed < 0.1  # Should be very fast for mock
            performance_timer.__init__()  # Reset timer

    def test_document_endpoints_response_time(self, mock_client, performance_timer):
        """Test document endpoints response times."""
        # Test upload
        performance_timer.start()
        upload_data = {
            "title": "Performance Test Doc",
            "content": "Content for performance testing",
            "document_type": "text"
        }
        response = mock_client.post("/api/v1/documents/upload", json=upload_data)
        performance_timer.stop()
        
        assert response.status_code == status.HTTP_200_OK
        assert performance_timer.elapsed < 0.1
        
        # Test list
        performance_timer.start()
        response = mock_client.get("/api/v1/documents/")
        performance_timer.stop()
        
        assert response.status_code == status.HTTP_200_OK
        assert performance_timer.elapsed < 0.1

    def test_search_response_time(self, mock_client, performance_timer):
        """Test search endpoint response time."""
        search_data = {
            "query": "performance test query",
            "limit": 10
        }
        
        performance_timer.start()
        response = mock_client.post("/api/v1/query/search", json=search_data)
        performance_timer.stop()
        
        assert response.status_code == status.HTTP_200_OK
        assert performance_timer.elapsed < 0.1

    def test_chunking_response_time(self, mock_client, performance_timer, sample_python_code):
        """Test chunking endpoint response time."""
        chunking_data = {
            "content": sample_python_code,
            "language": "python",
            "strategy": "hybrid"
        }
        
        performance_timer.start()
        response = mock_client.post("/api/v1/chunking/", json=chunking_data)
        performance_timer.stop()
        
        assert response.status_code == status.HTTP_200_OK
        assert performance_timer.elapsed < 0.1

    def test_system_endpoints_response_time(self, mock_client, performance_timer):
        """Test system endpoints response times."""
        endpoints = [
            "/api/v1/system/health",
            "/api/v1/system/status",
            "/api/v1/system/metrics",
            "/api/v1/system/logs",
            "/api/v1/system/config"
        ]
        
        for endpoint in endpoints:
            performance_timer.start()
            response = mock_client.get(endpoint)
            performance_timer.stop()
            
            assert response.status_code == status.HTTP_200_OK
            assert performance_timer.elapsed < 0.1
            performance_timer.__init__()  # Reset timer


class TestThroughput:
    """Test API throughput and concurrent request handling."""

    def test_concurrent_auth_requests(self, mock_client):
        """Test concurrent authentication requests."""
        def make_auth_request():
            login_data = {
                "email": "test@example.com",
                "password": "TestPassword123"
            }
            response = mock_client.post("/api/v1/auth/login", json=login_data)
            return response.status_code
        
        # Test with 10 concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_auth_request) for _ in range(10)]
            results = [future.result() for future in as_completed(futures)]
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 10

    def test_concurrent_document_uploads(self, mock_client):
        """Test concurrent document upload requests."""
        def upload_document(doc_id):
            doc_data = {
                "title": f"Concurrent Doc {doc_id}",
                "content": f"Content for document {doc_id}",
                "document_type": "text"
            }
            response = mock_client.post("/api/v1/documents/upload", json=doc_data)
            return response.status_code
        
        # Test with 15 concurrent uploads
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(upload_document, i) for i in range(15)]
            results = [future.result() for future in as_completed(futures)]
        
        assert all(status == 200 for status in results)
        assert len(results) == 15

    def test_concurrent_search_requests(self, mock_client):
        """Test concurrent search requests."""
        def make_search_request(query_id):
            search_data = {
                "query": f"concurrent search query {query_id}",
                "limit": 5
            }
            response = mock_client.post("/api/v1/query/search", json=search_data)
            return response.status_code
        
        # Test with 20 concurrent searches
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_search_request, i) for i in range(20)]
            results = [future.result() for future in as_completed(futures)]
        
        assert all(status == 200 for status in results)
        assert len(results) == 20

    def test_mixed_endpoint_concurrency(self, mock_client):
        """Test concurrent requests to different endpoints."""
        def make_mixed_requests():
            results = []
            
            # Auth request
            auth_response = mock_client.post("/api/v1/auth/login", json={
                "email": "test@example.com", "password": "Test123"
            })
            results.append(auth_response.status_code)
            
            # Document request
            doc_response = mock_client.get("/api/v1/documents/")
            results.append(doc_response.status_code)
            
            # Search request
            search_response = mock_client.post("/api/v1/query/search", json={
                "query": "mixed test", "limit": 3
            })
            results.append(search_response.status_code)
            
            # System request
            system_response = mock_client.get("/api/v1/system/health")
            results.append(system_response.status_code)
            
            return results
        
        # Test with 5 concurrent mixed request sets
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_mixed_requests) for _ in range(5)]
            all_results = []
            for future in as_completed(futures):
                all_results.extend(future.result())
        
        # All requests should succeed
        assert all(status == 200 for status in all_results)
        assert len(all_results) == 20  # 5 sets × 4 requests each


class TestScalability:
    """Test API scalability characteristics."""

    def test_increasing_load_pattern(self, mock_client):
        """Test API behavior under increasing load."""
        load_levels = [5, 10, 20, 30]
        results = {}
        
        for load in load_levels:
            start_time = time.time()
            
            def make_request():
                response = mock_client.get("/api/v1/system/health")
                return response.status_code
            
            with ThreadPoolExecutor(max_workers=load) as executor:
                futures = [executor.submit(make_request) for _ in range(load)]
                statuses = [future.result() for future in as_completed(futures)]
            
            end_time = time.time()
            duration = end_time - start_time
            
            results[load] = {
                "duration": duration,
                "success_rate": sum(1 for s in statuses if s == 200) / len(statuses),
                "requests_per_second": load / duration
            }
        
        # Verify all loads handled successfully
        for load, metrics in results.items():
            assert metrics["success_rate"] == 1.0  # 100% success rate
            assert metrics["requests_per_second"] > 0

    def test_sustained_load(self, mock_client):
        """Test API under sustained load."""
        duration_seconds = 5
        requests_per_second = 10
        total_requests = duration_seconds * requests_per_second
        
        start_time = time.time()
        results = []
        
        def make_sustained_request():
            response = mock_client.get("/api/v1/system/status")
            return response.status_code
        
        # Make requests at steady rate
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i in range(total_requests):
                future = executor.submit(make_sustained_request)
                futures.append(future)
                
                # Control rate
                if i % requests_per_second == 0 and i > 0:
                    time.sleep(0.1)  # Small delay to control rate
            
            results = [future.result() for future in as_completed(futures)]
        
        end_time = time.time()
        actual_duration = end_time - start_time
        
        # Verify performance
        success_rate = sum(1 for s in results if s == 200) / len(results)
        actual_rps = len(results) / actual_duration
        
        assert success_rate == 1.0  # 100% success rate
        assert actual_rps > 0
        assert len(results) == total_requests

    def test_burst_traffic_handling(self, mock_client):
        """Test API handling of burst traffic patterns."""
        # Simulate burst: many requests in short time, then quiet period
        burst_size = 50
        
        def burst_request():
            response = mock_client.post("/api/v1/query/search", json={
                "query": "burst test query",
                "limit": 5
            })
            return response.status_code
        
        # Create burst
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=burst_size) as executor:
            futures = [executor.submit(burst_request) for _ in range(burst_size)]
            results = [future.result() for future in as_completed(futures)]
        end_time = time.time()
        
        burst_duration = end_time - start_time
        
        # Verify burst handling
        success_rate = sum(1 for s in results if s == 200) / len(results)
        burst_rps = burst_size / burst_duration
        
        assert success_rate == 1.0  # All requests successful
        assert burst_rps > 0
        assert len(results) == burst_size


class TestMemoryAndResourceUsage:
    """Test memory usage and resource consumption patterns."""

    def test_memory_usage_during_large_operations(self, mock_client):
        """Test memory usage during large operations."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform large operations
        large_operations = []
        
        # Large document upload
        large_doc = {
            "title": "Large Document",
            "content": "Large content " * 1000,  # Simulate large content
            "document_type": "text"
        }
        response = mock_client.post("/api/v1/documents/upload", json=large_doc)
        large_operations.append(response.status_code)
        
        # Large search query
        search_data = {
            "query": "large search query with many terms and conditions",
            "limit": 50
        }
        response = mock_client.post("/api/v1/query/search", json=search_data)
        large_operations.append(response.status_code)
        
        # Large chunking operation
        large_code = "def function_{}():\n    pass\n" * 100
        chunking_data = {
            "content": large_code,
            "language": "python",
            "strategy": "hybrid"
        }
        response = mock_client.post("/api/v1/chunking/", json=chunking_data)
        large_operations.append(response.status_code)
        
        # Get final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Verify operations succeeded
        assert all(status == 200 for status in large_operations)
        
        # Memory increase should be reasonable (less than 100MB for mock operations)
        assert memory_increase < 100 * 1024 * 1024  # 100MB

    def test_resource_cleanup_after_operations(self, mock_client):
        """Test that resources are properly cleaned up after operations."""
        import gc
        
        # Force garbage collection before test
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Perform multiple operations
        for i in range(10):
            # Document operations
            mock_client.post("/api/v1/documents/upload", json={
                "title": f"Test Doc {i}",
                "content": f"Content {i}",
                "document_type": "text"
            })
            
            # Search operations
            mock_client.post("/api/v1/query/search", json={
                "query": f"test query {i}",
                "limit": 5
            })
            
            # System operations
            mock_client.get("/api/v1/system/health")
        
        # Force garbage collection after operations
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Object count should not increase dramatically
        object_increase = final_objects - initial_objects
        assert object_increase < 1000  # Reasonable increase for mock operations


class TestEndpointSpecificPerformance:
    """Test performance characteristics of specific endpoints."""

    def test_batch_upload_performance(self, mock_client, performance_timer):
        """Test batch upload performance vs individual uploads."""
        # Test individual uploads
        individual_times = []
        for i in range(5):
            performance_timer.start()
            response = mock_client.post("/api/v1/documents/upload", json={
                "title": f"Individual Doc {i}",
                "content": f"Content {i}",
                "document_type": "text"
            })
            performance_timer.stop()
            individual_times.append(performance_timer.elapsed)
            assert response.status_code == 200
            performance_timer.__init__()
        
        # Test batch upload
        batch_data = {
            "files": [
                {"title": f"Batch Doc {i}", "content": f"Content {i}"}
                for i in range(5)
            ]
        }
        
        performance_timer.start()
        response = mock_client.post("/api/v1/documents/upload/batch", json=batch_data)
        performance_timer.stop()
        batch_time = performance_timer.elapsed
        
        assert response.status_code == 200
        
        # Batch should be faster than sum of individual uploads
        total_individual_time = sum(individual_times)
        assert batch_time < total_individual_time

    def test_pagination_performance(self, mock_client, performance_timer):
        """Test pagination performance with different page sizes."""
        page_sizes = [5, 10, 20, 50]
        times = {}
        
        for page_size in page_sizes:
            performance_timer.start()
            response = mock_client.get(f"/api/v1/documents/?per_page={page_size}")
            performance_timer.stop()
            
            times[page_size] = performance_timer.elapsed
            assert response.status_code == 200
            performance_timer.__init__()
        
        # All pagination requests should be fast for mock
        for page_size, elapsed_time in times.items():
            assert elapsed_time < 0.1

    def test_complex_search_performance(self, mock_client, performance_timer):
        """Test performance of complex search queries."""
        complex_searches = [
            {
                "query": "simple query",
                "limit": 10
            },
            {
                "query": "complex query with multiple terms and filters",
                "collection_id": "col_test_1",
                "document_types": ["text", "markdown"],
                "tags": ["important", "recent"],
                "limit": 20
            },
            {
                "query": "very complex query with many parameters and conditions",
                "collection_id": "col_test_1",
                "document_types": ["text", "markdown", "code"],
                "tags": ["tag1", "tag2", "tag3", "tag4"],
                "limit": 50
            }
        ]
        
        for i, search_data in enumerate(complex_searches):
            performance_timer.start()
            response = mock_client.post("/api/v1/query/search", json=search_data)
            performance_timer.stop()
            
            assert response.status_code == 200
            assert performance_timer.elapsed < 0.1  # Should be fast for mock
            performance_timer.__init__()


@pytest.mark.slow
class TestLongRunningOperations:
    """Test long-running operations and timeouts."""

    def test_extended_load_test(self, mock_client):
        """Test extended load over longer period."""
        duration_minutes = 1  # 1 minute test
        requests_per_minute = 60
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        results = []
        request_count = 0
        
        while time.time() < end_time:
            response = mock_client.get("/api/v1/system/health")
            results.append(response.status_code)
            request_count += 1
            
            # Control rate
            if request_count % 10 == 0:
                time.sleep(0.1)
        
        # Verify sustained performance
        success_rate = sum(1 for s in results if s == 200) / len(results)
        actual_duration = time.time() - start_time
        actual_rps = len(results) / actual_duration
        
        assert success_rate >= 0.99  # At least 99% success rate
        assert actual_rps > 0
        assert len(results) > 0
