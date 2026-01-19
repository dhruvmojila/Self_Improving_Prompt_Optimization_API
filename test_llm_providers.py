"""
Test script for LLM providers.
Verifies Groq and Gemini connectivity with rate limiting.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.infrastructure.llm import LLMFactory, LLMConfig
from app.core.config import settings


def test_groq_provider():
    """Test Groq provider with llama model."""
    print("\n🧪 Testing Groq Provider...")
    print(f"Model: {settings.DEFAULT_TEACHER_MODEL}")
    
    try:
        provider = LLMFactory.create_provider(
            model_name="llama-3.1-8b-instant",
            provider="groq"
        )
        
        config = LLMConfig(temperature=0.7, max_tokens=100)
        
        response = provider.generate(
            prompt="Say 'Hello from Groq!' and nothing else.",
            config=config
        )
        
        print(f"✅ SUCCESS: {response.content[:100]}")
        print(f"   Tokens: {response.tokens_used}")
        print(f"   Provider: {response.provider}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_gemini_provider():
    """Test Gemini provider."""
    print("\n🧪 Testing Gemini Provider...")
    print(f"Model: {settings.DEFAULT_STUDENT_MODEL}")
    
    try:
        provider = LLMFactory.create_provider(
            model_name="gemini-2.5-flash-lite",
            provider="gemini"
        )
        
        config = LLMConfig(temperature=0.7, max_tokens=100)
        
        response = provider.generate(
            prompt="Say 'Hello from Gemini!' and nothing else.",
            config=config
        )
        
        print(f"✅ SUCCESS: {response.content[:100]}")
        print(f"   Provider: {response.provider}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_factory_roles():
    """Test LLM factory role selection."""
    print("\n🧪 Testing LLM Factory Roles...")
    
    try:
        teacher = LLMFactory.create_teacher()
        student = LLMFactory.create_student()
        judge = LLMFactory.create_judge()
        
        print(f"✅ Teacher: {teacher.model_name} ({teacher.provider_name})")
        print(f"✅ Student: {student.model_name} ({student.provider_name})")
        print(f"✅ Judge: {judge.model_name} ({judge.provider_name})")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def main():
    """Run all LLM provider tests."""
    print("=" * 60)
    print("LLM Provider Integration Tests")
    print("=" * 60)
    
    print(f"\n📋 Configuration:")
    print(f"   Groq API Key: {settings.GROQ_API_KEY[:10]}...")
    print(f"   Gemini API Key: {settings.GEMINI_API_KEY[:10]}...")
    print(f"   Groq RPM Limit: {settings.GROQ_RPM_LIMIT}")
    print(f"   Gemini RPM Limit: {settings.GEMINI_RPM_LIMIT}")
    
    results = []
    
    # Test factory roles first
    results.append(("Factory Roles", test_factory_roles()))
    
    # Test Groq
    results.append(("Groq Provider", test_groq_provider()))
    
    # Test Gemini
    results.append(("Gemini Provider", test_gemini_provider()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! LLM infrastructure is ready.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check API keys and rate limits.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
