"""
End-to-End Test Suite for Production System.
Tests the complete flow with Celery workers and API endpoints.
"""

import asyncio
import time
import requests
from datetime import datetime

# Test with API running on localhost:8000
BASE_URL = "http://localhost:8000"


def test_complete_workflow():
    """Test the complete optimization workflow."""
    
    print("=" * 70)
    print("END-TO-END PRODUCTION TEST")
    print("=" * 70)
    
    # Step 1: Health check
    print("\n1. Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200, "Health check failed"
    print(f"✓ API is healthy: {response.json()}")
    
    # Step 2: Upload dataset
    print("\n2. Uploading dataset...")
    
    # Create sample CSV data
    import io
    csv_data = """text,sentiment
I love this product with all my heart and soul!,positive
This is absolutely terrible and I hate it completely.,negative
It's okay I guess could be better or worse.,neutral
Best purchase ever made in my entire life!,positive
Complete waste of my hard earned money.,negative
Decent quality for what you actually pay for.,neutral"""
    
    files = {
        'file': ('sentiment_data.csv', io.BytesIO(csv_data.encode()), 'text/csv')
    }
    data = {
        'name': 'E2E Sentiment Test',
        'input_fields': '["text"]',
        'output_fields': '["sentiment"]',
        'train_split': 0.5,
        'dev_split': 0.33,
        'test_split': 0.17
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/datasets/upload",
        files=files,
        data=data
    )
    
    if response.status_code != 201:
        print(f"✗ Dataset upload failed: {response.text}")
        return
    
    dataset = response.json()
    dataset_id = dataset['id']
    print(f"✓ Dataset uploaded: {dataset_id}")
    print(f"  Total rows: {dataset['total_rows']}")
    
    # Step 3: Create baseline prompt
    print("\n3. Creating baseline prompt...")
    
    prompt_data = {
        'name': 'E2E Baseline Classifier',
        'template': 'Classify sentiment of: {text}',
        'signature': {
            'inputs': ['text'],
            'outputs': ['sentiment']
        },
        'description': 'End-to-end test prompt',
        'tags': ['e2e-test']
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/prompts",
        json=prompt_data
    )
    
    if response.status_code != 201:
        print(f"✗ Prompt creation failed: {response.text}")
        return
    
    prompt = response.json()
    prompt_id = prompt['id']
    print(f"✓ Prompt created: {prompt_id}")
    
    # Step 4: Start optimization job
    print("\n4. Starting optimization job...")
    
    opt_config = {
        'prompt_id': prompt_id,
        'dataset_id': dataset_id,
        'config': {
            'teacher_model': 'llama-3.3-70b-versatile',
            'student_model': 'llama-3.1-8b-instant',
            'teacher_provider': 'groq',
            'student_provider': 'groq',
            'optimizer_type': 'bootstrap',
            'num_trials': 3,
            'num_fewshot_examples': 2,
            'budget_usd': 0.5,
            'metric_name': 'correctness_metric'
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/optimization/start",
        json=opt_config
    )
    
    if response.status_code != 201:
        print(f"✗ Job creation failed: {response.text}")
        return
    
    job = response.json()
    job_id = job['id']
    print(f"✓ Job created: {job_id}")
    print(f"  Status: {job['status']}")
    
    # Step 5: Poll job status
    print("\n5. Polling job status (max 5 minutes)...")
    
    max_wait = 300  # 5 minutes
    poll_interval = 5  # 5 seconds
    elapsed = 0
    
    while elapsed < max_wait:
        response = requests.get(f"{BASE_URL}/api/v1/optimization/jobs/{job_id}")
        
        if response.status_code != 200:
            print(f"✗ Status check failed: {response.text}")
            break
        
        job_status = response.json()
        print(job_status)
        status = job_status['status']
        progress = job_status.get('progress', {})
        progress_pct = progress.get('progress_pct', 0)
        message = progress.get('message', '')
        
        print(f"  [{int(progress_pct):3d}%] {status}: {message}")
        
        if status == 'completed':
            print("\n✓ Job completed successfully!")
            break
        elif status == 'failed':
            print("\n✗ Job failed")
            break
        
        time.sleep(poll_interval)
        elapsed += poll_interval
    
    if elapsed >= max_wait:
        print("\n⚠️  Job timed out")
        return
    
    # Step 6: Get results
    if status == 'completed':
        print("\n6. Fetching optimization results...")
        
        response = requests.get(
            f"{BASE_URL}/api/v1/optimization/jobs/{job_id}/result"
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n" + "=" * 70)
            print("OPTIMIZATION RESULTS")
            print("=" * 70)
            print(f"Baseline score:    {result.get('baseline_score', 'N/A')}")
            print(f"Optimized score:   {result.get('optimized_score', 'N/A')}")
            print(f"Improvement:       {result.get('improvement_pct', 'N/A'):+.1f}%")
            print(f"Few-shot examples: {result.get('num_examples', 'N/A')}")
            print(f"Optimizer:         {result.get('optimizer_type', 'N/A')}")
            print(f"Student model:     {result['model_info']['student']}")
            print(f"Teacher model:     {result['model_info']['teacher']}")
            print("=" * 70)
            
            # Step 7: Promote to production (optional)
            print("\n7. Testing promotion endpoint...")
            
            response = requests.post(
                f"{BASE_URL}/api/v1/optimization/jobs/{job_id}/promote",
                json={'notes': 'E2E test promotion'}
            )
            
            if response.status_code == 200:
                promotion = response.json()
                print(f"✓ Promoted: {promotion['message']}")
            else:
                print(f"⚠️  Promotion skipped: {response.text}")
        
        print("\n" + "=" * 70)
        print("✅ END-TO-END TEST PASSED!")
        print("=" * 70)


def test_celery_worker():
    """Test Celery worker health."""
    print("\nTesting Celery worker...")
    
    try:
        from app.workers.tasks import health_check
        
        result = health_check.delay()
        response = result.get(timeout=10)
        
        print(f"✓ Celery worker is healthy: {response}")
        return True
    except Exception as e:
        print(f"✗ Celery test failed: {e}")
        print("  Make sure Redis and Celery worker are running!")
        return False


if __name__ == "__main__":
    print("\nPRE-REQUISITES:")
    print("1. Start API: uvicorn app.api.main:app --reload --port 8000")
    print("2. Start Redis: redis-server (or Docker)")
    print("3. Start Celery: celery -A app.workers.celery_app worker -l info")
    print("\nPress Enter when ready...")
    input()
    
    # Test Celery first
    if test_celery_worker():
        print("\n")
        # Run full E2E test
        test_complete_workflow()
    else:
        print("\n⚠️  Skipping E2E test - fix Celery first")
