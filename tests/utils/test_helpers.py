"""Test helper functions for Cognify tests."""

import time
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from contextlib import asynccontextmanager

from app.services.chunking.base import ChunkingRequest, ChunkingResponse


def create_test_file(content: str, suffix: str = ".py") -> Path:
    """Create a temporary test file with the given content."""
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        f.write(content)
        return Path(f.name)


def generate_test_code(language: str = "python", complexity: str = "simple") -> str:
    """Generate test code for different languages and complexity levels."""
    
    if language == "python":
        if complexity == "simple":
            return '''
def hello_world():
    """Simple hello world function."""
    return "Hello, World!"

def add_numbers(a, b):
    """Add two numbers."""
    return a + b

if __name__ == "__main__":
    print(hello_world())
    print(add_numbers(5, 3))
'''
        elif complexity == "medium":
            return '''
class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a, b):
        """Subtract two numbers."""
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result
    
    def multiply(self, a, b):
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def divide(self, a, b):
        """Divide two numbers."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result
    
    def get_history(self):
        """Get calculation history."""
        return self.history.copy()
    
    def clear_history(self):
        """Clear calculation history."""
        self.history.clear()

def fibonacci(n):
    """Calculate fibonacci number recursively."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def fibonacci_iterative(n):
    """Calculate fibonacci number iteratively."""
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def main():
    """Main function demonstrating calculator usage."""
    calc = Calculator()
    
    # Basic operations
    print(f"5 + 3 = {calc.add(5, 3)}")
    print(f"10 - 4 = {calc.subtract(10, 4)}")
    print(f"6 * 7 = {calc.multiply(6, 7)}")
    print(f"15 / 3 = {calc.divide(15, 3)}")
    
    # Fibonacci calculations
    print(f"Fibonacci(10) recursive: {fibonacci(10)}")
    print(f"Fibonacci(10) iterative: {fibonacci_iterative(10)}")
    
    # History
    print("Calculation history:")
    for entry in calc.get_history():
        print(f"  {entry}")

if __name__ == "__main__":
    main()
'''
        else:  # complex
            return '''
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Task:
    """Represents a computational task."""
    id: str
    name: str
    priority: int
    data: Dict[str, Any]
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class TaskProcessor(ABC):
    """Abstract base class for task processors."""
    
    @abstractmethod
    async def process(self, task: Task) -> Dict[str, Any]:
        """Process a task and return results."""
        pass
    
    @abstractmethod
    def can_process(self, task: Task) -> bool:
        """Check if this processor can handle the task."""
        pass

class MathTaskProcessor(TaskProcessor):
    """Processor for mathematical tasks."""
    
    async def process(self, task: Task) -> Dict[str, Any]:
        """Process mathematical tasks."""
        operation = task.data.get('operation')
        operands = task.data.get('operands', [])
        
        if operation == 'add':
            result = sum(operands)
        elif operation == 'multiply':
            result = 1
            for operand in operands:
                result *= operand
        elif operation == 'fibonacci':
            n = operands[0] if operands else 0
            result = await self._fibonacci_async(n)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
        
        return {
            'task_id': task.id,
            'result': result,
            'operation': operation,
            'operands': operands
        }
    
    def can_process(self, task: Task) -> bool:
        """Check if this is a math task."""
        return task.data.get('type') == 'math'
    
    async def _fibonacci_async(self, n: int) -> int:
        """Calculate fibonacci asynchronously."""
        if n <= 1:
            return n
        
        # Simulate async computation
        await asyncio.sleep(0.001)
        
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

class TaskScheduler:
    """Schedules and executes tasks with dependency management."""
    
    def __init__(self):
        self.processors: List[TaskProcessor] = []
        self.completed_tasks: Dict[str, Dict[str, Any]] = {}
        self.pending_tasks: List[Task] = []
    
    def add_processor(self, processor: TaskProcessor):
        """Add a task processor."""
        self.processors.append(processor)
    
    def add_task(self, task: Task):
        """Add a task to the scheduler."""
        self.pending_tasks.append(task)
    
    async def execute_all(self) -> Dict[str, Dict[str, Any]]:
        """Execute all pending tasks respecting dependencies."""
        results = {}
        
        while self.pending_tasks:
            # Find tasks with satisfied dependencies
            ready_tasks = [
                task for task in self.pending_tasks
                if all(dep in self.completed_tasks for dep in task.dependencies)
            ]
            
            if not ready_tasks:
                raise RuntimeError("Circular dependency detected or missing dependencies")
            
            # Process ready tasks
            for task in ready_tasks:
                processor = self._find_processor(task)
                if processor:
                    logger.info(f"Processing task {task.id}: {task.name}")
                    result = await processor.process(task)
                    results[task.id] = result
                    self.completed_tasks[task.id] = result
                    self.pending_tasks.remove(task)
                else:
                    raise RuntimeError(f"No processor found for task {task.id}")
        
        return results
    
    def _find_processor(self, task: Task) -> Optional[TaskProcessor]:
        """Find a suitable processor for the task."""
        for processor in self.processors:
            if processor.can_process(task):
                return processor
        return None

async def main():
    """Main function demonstrating the task scheduler."""
    # Create scheduler and add processor
    scheduler = TaskScheduler()
    scheduler.add_processor(MathTaskProcessor())
    
    # Create tasks
    tasks = [
        Task(
            id="task1",
            name="Add numbers",
            priority=1,
            data={'type': 'math', 'operation': 'add', 'operands': [1, 2, 3, 4, 5]}
        ),
        Task(
            id="task2", 
            name="Multiply numbers",
            priority=2,
            data={'type': 'math', 'operation': 'multiply', 'operands': [2, 3, 4]}
        ),
        Task(
            id="task3",
            name="Calculate fibonacci",
            priority=3,
            data={'type': 'math', 'operation': 'fibonacci', 'operands': [15]},
            dependencies=["task1"]  # Depends on task1
        )
    ]
    
    # Add tasks to scheduler
    for task in tasks:
        scheduler.add_task(task)
    
    # Execute all tasks
    try:
        results = await scheduler.execute_all()
        
        print("Task execution results:")
        for task_id, result in results.items():
            print(f"  {task_id}: {result}")
            
    except Exception as e:
        logger.error(f"Task execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    elif language == "javascript":
        return '''
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

class Calculator {
    constructor() {
        this.history = [];
    }
    
    add(a, b) {
        const result = a + b;
        this.history.push(`${a} + ${b} = ${result}`);
        return result;
    }
    
    multiply(a, b) {
        const result = a * b;
        this.history.push(`${a} * ${b} = ${result}`);
        return result;
    }
    
    getHistory() {
        return [...this.history];
    }
}

export { fibonacci, Calculator };
'''
    
    else:
        return f"// Sample {language} code\nfunction main() {{\n    console.log('Hello from {language}');\n}}"


def measure_performance(func):
    """Decorator to measure function performance."""
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        
        # Add performance metadata to result if it's a response object
        if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
            result.metadata['performance'] = {
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time
            }
        
        return result
    
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        return result, end_time - start_time
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


async def compare_chunking_strategies(
    content: str,
    language: str,
    file_path: str,
    chunking_service
) -> Dict[str, ChunkingResponse]:
    """Compare different chunking strategies on the same content."""
    strategies = ['agentic', 'ast']
    results = {}
    
    for strategy in strategies:
        request = ChunkingRequest(
            content=content,
            language=language,
            file_path=file_path,
            purpose="general",
            force_agentic=(strategy == 'agentic')
        )
        
        start_time = time.time()
        response = await chunking_service.chunk_content(request)
        end_time = time.time()
        
        # Add timing information
        if hasattr(response, 'metadata'):
            response.metadata['strategy_timing'] = end_time - start_time
        
        results[strategy] = response
    
    return results


def create_chunking_request(
    content: Optional[str] = None,
    language: str = "python",
    file_path: str = "test.py",
    purpose: str = "general",
    **kwargs
) -> ChunkingRequest:
    """Create a chunking request with default values."""
    if content is None:
        content = generate_test_code(language, "simple")
    
    return ChunkingRequest(
        content=content,
        language=language,
        file_path=file_path,
        purpose=purpose,
        **kwargs
    )


@asynccontextmanager
async def temporary_service(service_class, *args, **kwargs):
    """Context manager for temporary service instances."""
    service = service_class(*args, **kwargs)
    try:
        if hasattr(service, 'initialize'):
            await service.initialize()
        yield service
    finally:
        if hasattr(service, 'cleanup'):
            await service.cleanup()


def count_lines(content: str) -> int:
    """Count the number of lines in content."""
    return len(content.strip().split('\n'))


def extract_functions(content: str) -> List[str]:
    """Extract function names from Python code (simple regex-based)."""
    import re
    pattern = r'def\s+(\w+)\s*\('
    return re.findall(pattern, content)


def extract_classes(content: str) -> List[str]:
    """Extract class names from Python code (simple regex-based)."""
    import re
    pattern = r'class\s+(\w+)\s*[:\(]'
    return re.findall(pattern, content)


def calculate_complexity_score(content: str) -> float:
    """Calculate a simple complexity score based on content."""
    lines = content.strip().split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    
    # Simple metrics
    total_lines = len(non_empty_lines)
    function_count = len(extract_functions(content))
    class_count = len(extract_classes(content))
    
    # Basic complexity calculation
    complexity = (total_lines * 0.1) + (function_count * 2) + (class_count * 5)
    
    # Normalize to 0-1 scale (assuming max complexity of 100)
    return min(complexity / 100.0, 1.0)


def validate_chunk_boundaries(chunks: List[Dict[str, Any]], total_lines: int) -> List[str]:
    """Validate chunk boundaries and return list of issues."""
    issues = []
    
    if not chunks:
        return ["No chunks generated"]
    
    # Check for overlapping chunks
    sorted_chunks = sorted(chunks, key=lambda c: c.get('start_line', 0))
    
    for i in range(len(sorted_chunks) - 1):
        current_end = sorted_chunks[i].get('end_line', 0)
        next_start = sorted_chunks[i + 1].get('start_line', 0)
        
        if current_end >= next_start:
            issues.append(f"Chunks {i} and {i+1} overlap")
    
    # Check coverage
    first_start = sorted_chunks[0].get('start_line', 1)
    last_end = sorted_chunks[-1].get('end_line', 0)
    
    if first_start > 5:
        issues.append(f"First chunk starts too late (line {first_start})")
    
    if last_end < total_lines - 5:
        issues.append(f"Last chunk ends too early (line {last_end}, total: {total_lines})")
    
    return issues
