"""
LLM Provider abstraction layer.
Supports Groq and Gemini with unified interface, rate limiting, and retry logic.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import time

from groq import Groq
from google import genai
from google.genai import types
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.core.security import retry_with_exponential_backoff, check_rate_limit, get_rate_limiter


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None


@dataclass
class LLMConfig:
    """Configuration for LLM requests."""
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    stop: Optional[List[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    All providers must implement the generate method.
    """
    
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        self.provider_name = "base"
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None,
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate completion from prompt."""
        pass
    
    @abstractmethod
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Generate completion from chat messages."""
        pass
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Simple approximation: ~4 characters per token
        return len(text) // 4


class GroqProvider(LLMProvider):
    """Groq LLM provider with rate limiting and retry logic."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        super().__init__(model_name, api_key or settings.GROQ_API_KEY)
        self.provider_name = "groq"
        self.client = Groq(api_key=self.api_key)
        self.rate_limiter = get_rate_limiter("groq")
    
    @retry_with_exponential_backoff(provider="groq")
    def generate(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None,
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate completion with Groq."""
        config = config or LLMConfig()
        
        # Check rate limit
        check_rate_limit(self.model_name, provider="groq")
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Make request
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            stop=config.stop,
            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty,
        )
        
        # Extract response
        content = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else None
        
        return LLMResponse(
            content=content,
            model=self.model_name,
            provider=self.provider_name,
            tokens_used=tokens_used,
            cost_usd=0.0,  # Groq is free tier
            finish_reason=response.choices[0].finish_reason,
            raw_response=response
        )
    
    @retry_with_exponential_backoff(provider="groq")
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Generate completion from chat messages."""
        config = config or LLMConfig()
        
        # Check rate limit
        check_rate_limit(self.model_name, provider="groq")
        
        # Make request
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            stop=config.stop,
        )
        
        content = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else None
        
        return LLMResponse(
            content=content,
            model=self.model_name,
            provider=self.provider_name,
            tokens_used=tokens_used,
            cost_usd=0.0,
            finish_reason=response.choices[0].finish_reason,
            raw_response=response
        )
    
    def get_langchain_model(self, config: Optional[LLMConfig] = None):
        """Get LangChain-compatible model instance."""
        config = config or LLMConfig()
        return ChatGroq(
            model=self.model_name,
            groq_api_key=self.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider using new google.genai SDK."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        super().__init__(model_name, api_key or settings.GEMINI_API_KEY)
        self.provider_name = "gemini"
        self.client = genai.Client(api_key=self.api_key)
        self.rate_limiter = get_rate_limiter("gemini")

    @retry_with_exponential_backoff(provider="gemini")
    def generate(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None,
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        config = config or LLMConfig()

        check_rate_limit(self.model_name, provider="gemini")

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=config.temperature,
                top_p=config.top_p,
                max_output_tokens=config.max_tokens,
                stop_sequences=config.stop,
            ),
        )

        return LLMResponse(
            content=response.text,
            model=self.model_name,
            provider=self.provider_name,
            tokens_used=None,
            cost_usd=0.0,
            raw_response=response,
        )
    
    @retry_with_exponential_backoff(provider="gemini")
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Generate completion from chat messages using Gemini."""
        config = config or LLMConfig()

        check_rate_limit(self.model_name, provider="gemini")

        # Convert messages into a single prompt (Gemini does not require role tokens)
        system_prompt = ""
        user_prompt = ""

        for msg in messages:
            if msg["role"] == "system":
                system_prompt += msg["content"] + "\n"
            elif msg["role"] == "user":
                user_prompt += msg["content"] + "\n"

        full_prompt = (
            f"{system_prompt.strip()}\n\n{user_prompt.strip()}"
            if system_prompt
            else user_prompt.strip()
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=config.temperature,
                top_p=config.top_p,
                max_output_tokens=config.max_tokens,
                stop_sequences=config.stop,
            ),
        )

        return LLMResponse(
            content=response.text,
            model=self.model_name,
            provider=self.provider_name,
            tokens_used=None,
            cost_usd=0.0,
            raw_response=response,
        )
    
    def get_langchain_model(self, config: Optional[LLMConfig] = None):
        """Get LangChain-compatible model instance."""
        config = config or LLMConfig()
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
        )


class LLMFactory:
    """
    Factory for creating LLM provider instances.
    Handles model selection for teacher/student/judge roles.
    """
    
    @staticmethod
    def create_provider(
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        role: str = "teacher"
    ) -> LLMProvider:
        """
        Create an LLM provider instance.
        
        Args:
            model_name: Specific model name (overrides role defaults)
            provider: Provider name ("groq" or "gemini")
            role: Model role ("teacher", "student", or "judge")
            
        Returns:
            LLMProvider instance
        """
        # Determine provider and model from role if not specified
        if model_name is None or provider is None:
            if role == "teacher":
                provider = provider or settings.DEFAULT_TEACHER_PROVIDER
                model_name = model_name or settings.DEFAULT_TEACHER_MODEL
            elif role == "student":
                provider = provider or settings.DEFAULT_STUDENT_PROVIDER
                model_name = model_name or settings.DEFAULT_STUDENT_MODEL
            elif role == "judge":
                provider = provider or settings.DEFAULT_JUDGE_PROVIDER
                model_name = model_name or settings.DEFAULT_JUDGE_MODEL
            else:
                raise ValueError(f"Unknown role: {role}")
        
        # Create provider
        if provider.lower() == "groq":
            return GroqProvider(model_name=model_name)
        elif provider.lower() == "gemini":
            return GeminiProvider(model_name=model_name)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @staticmethod
    def create_teacher() -> LLMProvider:
        """Create teacher model (more capable, for optimization)."""
        return LLMFactory.create_provider(role="teacher")
    
    @staticmethod
    def create_student() -> LLMProvider:
        """Create student model (efficient, for inference)."""
        return LLMFactory.create_provider(role="student")
    
    @staticmethod
    def create_judge() -> LLMProvider:
        """Create judge model (for evaluation)."""
        return LLMFactory.create_provider(role="judge")
