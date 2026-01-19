"""
Security utilities for the Prompt Optimization API.
Includes API key validation, rate limiting, and authentication helpers.
"""

from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from tenacity import retry, stop_after_attempt, wait_exponential
import time
from collections import defaultdict
from threading import Lock
from app.core.config import settings


# API Key Header for authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# Simple in-memory rate limiter
class RateLimiter:
    """
    Simple token bucket rate limiter for API endpoints.
    Thread-safe implementation with per-key tracking.
    """
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.tokens: dict[str, list[float]] = defaultdict(list)
        self.lock = Lock()
    
    def is_allowed(self, key: str) -> bool:
        """
        Check if a request is allowed for the given key.
        
        Args:
            key: Identifier for the rate limit (e.g., API key, IP address)
            
        Returns:
            True if request is allowed, False if rate limit exceeded
        """
        with self.lock:
            now = time.time()
            window_start = now - 60  # 1 minute window
            
            # Remove expired tokens
            self.tokens[key] = [t for t in self.tokens[key] if t > window_start]
            
            # Check if we have capacity
            if len(self.tokens[key]) < self.requests_per_minute:
                self.tokens[key].append(now)
                return True
            else:
                return False
    
    def get_remaining(self, key: str) -> int:
        """Get remaining requests for the current window."""
        with self.lock:
            now = time.time()
            window_start = now - 60
            self.tokens[key] = [t for t in self.tokens[key] if t > window_start]
            return max(0, self.requests_per_minute - len(self.tokens[key]))


# Global rate limiter instances
groq_rate_limiter = RateLimiter(requests_per_minute=settings.GROQ_RPM_LIMIT)
gemini_rate_limiter = RateLimiter(requests_per_minute=settings.GEMINI_RPM_LIMIT)
api_rate_limiter = RateLimiter(requests_per_minute=60)  # General API rate limit


def get_rate_limiter(provider: str) -> RateLimiter:
    """Get the appropriate rate limiter for a provider."""
    if provider.lower() == "groq":
        return groq_rate_limiter
    elif provider.lower() == "gemini":
        return gemini_rate_limiter
    else:
        return api_rate_limiter


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """
    Verify API key from request header.
    
    For MVP, this is optional - can be disabled by setting api_key to None.
    In production, implement proper API key database validation.
    
    Args:
        api_key: API key from request header
        
    Returns:
        Validated API key
        
    Raises:
        HTTPException: If API key is invalid or missing
    """
    # For MVP: Skip validation (can be enabled later)
    # In production: Check against database of valid keys
    if api_key is None:
        # Allow requests without API key for development
        return "development"
    
    # TODO: Implement proper API key validation
    # For now, accept any non-empty key
    if len(api_key) < 10:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return api_key


def check_rate_limit(key: str, provider: str = "api") -> None:
    """
    Check rate limit for a given key and provider.
    
    Args:
        key: Identifier for rate limiting
        provider: Provider name ("groq", "gemini", or "api")
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    limiter = get_rate_limiter(provider)
    
    if not limiter.is_allowed(key):
        remaining_time = 60 - (time.time() % 60)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for {provider}. Try again in {int(remaining_time)}s",
            headers={"Retry-After": str(int(remaining_time))}
        )


# Retry decorator for LLM calls with exponential backoff
def retry_with_exponential_backoff(provider: str = "groq"):
    """
    Decorator factory for retrying LLM calls with exponential backoff.
    
    Args:
        provider: LLM provider name for rate limit checking
        
    Returns:
        Configured retry decorator
    """
    return retry(
        stop=stop_after_attempt(settings.MAX_RETRIES),
        wait=wait_exponential(
            min=settings.RETRY_MIN_WAIT_SECONDS,
            max=settings.RETRY_MAX_WAIT_SECONDS
        ),
        reraise=True
    )


# JWT token utilities (for future implementation)
class JWTManager:
    """
    JWT token manager for user authentication.
    Currently a placeholder - implement when user auth is needed.
    """
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[int] = None) -> str:
        """Create JWT access token."""
        # TODO: Implement JWT token creation
        raise NotImplementedError("JWT authentication not yet implemented")
    
    @staticmethod
    def verify_token(token: str) -> dict:
        """Verify and decode JWT token."""
        # TODO: Implement JWT token verification
        raise NotImplementedError("JWT authentication not yet implemented")
