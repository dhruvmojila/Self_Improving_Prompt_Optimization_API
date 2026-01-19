"""
Shared dependencies for API endpoints.
Provides dependency injection for common functionality.
"""

from typing import Optional
from fastapi import Depends, HTTPException, status

from app.core.security import verify_api_key, check_rate_limit
from app.infrastructure.pixeltable import get_pixeltable_client, PixeltableClient


def get_db() -> PixeltableClient:
    """
    Get Pixeltable client dependency.
    
    Returns:
        PixeltableClient instance
    """
    return get_pixeltable_client()


def get_current_user(api_key: str = Depends(verify_api_key)) -> str:
    """
    Get current authenticated user.
    For MVP, returns api_key as user_id.
    
    Args:
        api_key: Validated API key
        
    Returns:
        User ID (api_key for MVP)
    """
    return api_key


def check_api_rate_limit(api_key: str = Depends(verify_api_key)):
    """
    Check API rate limit for user.
    
    Args:
        api_key: User's API key
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    check_rate_limit(api_key, provider="api")
