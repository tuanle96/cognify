"""
Base agent classes for Cognify agentic system.

Provides foundation for LLM-powered agents that can make intelligent decisions
about code chunking, analysis, and optimization.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import structlog

from app.core.config import get_settings
from app.services.llm.service import llm_service


logger = structlog.get_logger(__name__)


# Simple message classes for agents
class LLMMessage:
    """Simple LLM message class."""
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class BaseAgent(ABC):
    """
    Base class for all Cognify agents.

    Provides common functionality for LLM-powered agents including:
    - LLM service integration
    - Task execution
    - Error handling and fallbacks
    - Logging and monitoring
    """

    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str,
        max_iterations: int = 3,
        temperature: float = 0.1,
        verbose: bool = False
    ):
        """
        Initialize base agent.

        Args:
            role: Agent's role description
            goal: Agent's primary goal
            backstory: Agent's background and expertise
            max_iterations: Maximum retry attempts
            temperature: LLM temperature for responses
            verbose: Enable verbose logging
        """
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.verbose = verbose

        # Initialize logger
        self.logger = structlog.get_logger(f"agent.{self.__class__.__name__}")

        # LLM service will be initialized lazily
        self._llm_service = None
        self._initialized = False

        # Performance tracking
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0

    async def initialize(self) -> None:
        """Initialize the agent and its dependencies."""
        if self._initialized:
            return

        try:
            self._llm_service = llm_service
            await self._llm_service.initialize()
            self._initialized = True

            if self.verbose:
                self.logger.info("Agent initialized", role=self.role)

        except Exception as e:
            self.logger.error("Failed to initialize agent", error=str(e))
            raise

    async def execute_task(self, task_description: str, context: Dict[str, Any] = None) -> str:
        """
        Execute a task using the LLM service.

        Args:
            task_description: Detailed description of the task
            context: Additional context for the task

        Returns:
            LLM response as string
        """
        if not self._initialized:
            await self.initialize()

        self.execution_count += 1

        # Prepare messages
        system_message = self._create_system_message()
        user_message = self._create_user_message(task_description, context)

        messages = [system_message, user_message]

        # Execute with retry logic
        for attempt in range(self.max_iterations):
            try:
                if self.verbose:
                    self.logger.info(
                        "Executing task",
                        attempt=attempt + 1,
                        task_preview=task_description[:100] + "..."
                    )

                # Use cached LLM generation for performance optimization
                try:
                    from app.services.cache.llm_cache import cached_llm_generate
                    response = await cached_llm_generate(
                        self._llm_service,
                        messages,
                        use_cache=True,
                        temperature=self.temperature,
                        max_tokens=12000
                    )
                except ImportError:
                    # Fallback to direct generation if cache not available
                    # Convert LLMMessage objects to dict format for LiteLLM
                    message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]

                    response_content = await self._llm_service.chat_completion(
                        messages=message_dicts,
                        temperature=self.temperature,
                        max_tokens=12000
                    )

                # Handle response (either from cache or direct call)
                if hasattr(response, 'content'):
                    # Response from cache (has .content attribute)
                    content = response.content
                else:
                    # Response from direct LiteLLM call (is string)
                    content = response_content if 'response_content' in locals() else response

                if content and content.strip():
                    self.success_count += 1

                    if self.verbose:
                        self.logger.info(
                            "Task completed successfully",
                            response_length=len(content)
                        )

                    return content.strip()
                else:
                    error_msg = "Empty response from LLM service"
                    error_msg += f" (content is: {repr(content)})"
                    raise Exception(error_msg)

            except Exception as e:
                self.logger.warning(
                    "Task execution failed",
                    attempt=attempt + 1,
                    error=str(e)
                )

                if attempt == self.max_iterations - 1:
                    self.failure_count += 1
                    raise Exception(f"Task failed after {self.max_iterations} attempts: {e}")

                # Wait before retry
                await asyncio.sleep(1.0 * (attempt + 1))

        raise Exception("Task execution failed")

    def _create_system_message(self) -> LLMMessage:
        """Create system message with agent's role and capabilities."""
        system_content = f"""You are a {self.role}.

GOAL: {self.goal}

BACKGROUND: {self.backstory}

INSTRUCTIONS:
- Provide detailed, accurate analysis based on your expertise
- Use structured output formats when requested (JSON, etc.)
- Explain your reasoning clearly
- Be precise and actionable in your recommendations
- Focus on the specific task requirements

RESPONSE FORMAT:
- Start with a brief summary of your analysis
- Provide detailed findings with clear reasoning
- End with specific recommendations or structured output as requested
- Use proper formatting for code, JSON, or other structured data
"""

        return LLMMessage(role="system", content=system_content)

    def _create_user_message(self, task_description: str, context: Dict[str, Any] = None) -> LLMMessage:
        """Create user message with task description and context."""
        content = f"TASK:\n{task_description}"

        if context:
            content += f"\n\nCONTEXT:\n{self._format_context(context)}"

        return LLMMessage(role="user", content=content)

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context dictionary for inclusion in prompt."""
        formatted_lines = []
        for key, value in context.items():
            if isinstance(value, (dict, list)):
                import json
                formatted_lines.append(f"{key}: {json.dumps(value, indent=2)}")
            else:
                formatted_lines.append(f"{key}: {value}")

        return "\n".join(formatted_lines)

    async def cleanup(self) -> None:
        """Cleanup agent resources."""
        if self._llm_service:
            await self._llm_service.cleanup()

        if self.verbose:
            self.logger.info(
                "Agent cleanup completed",
                executions=self.execution_count,
                successes=self.success_count,
                failures=self.failure_count
            )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics."""
        success_rate = (self.success_count / self.execution_count) if self.execution_count > 0 else 0.0

        return {
            "role": self.role,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": success_rate,
            "initialized": self._initialized
        }


class Task:
    """
    Represents a task to be executed by an agent.

    This is a simple data structure that holds task information
    for compatibility with existing code.
    """

    def __init__(
        self,
        description: str,
        agent: BaseAgent,
        expected_output: str = None,
        context: Dict[str, Any] = None
    ):
        """
        Initialize task.

        Args:
            description: Task description
            agent: Agent that will execute the task
            expected_output: Description of expected output format
            context: Additional context for the task
        """
        self.description = description
        self.agent = agent
        self.expected_output = expected_output
        self.context = context or {}

    async def execute(self) -> str:
        """Execute the task using the assigned agent."""
        full_description = self.description

        if self.expected_output:
            full_description += f"\n\nEXPECTED OUTPUT: {self.expected_output}"

        return await self.agent.execute_task(full_description, self.context)


# Compatibility aliases for existing code
Agent = BaseAgent


async def execute_async(task: Task) -> str:
    """
    Execute a task asynchronously.

    This function provides compatibility with existing agent code
    that expects an execute_async function.
    """
    return await task.execute()
