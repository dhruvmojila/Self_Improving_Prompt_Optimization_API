# 🚀 Deployment & Testing Guide

## Production Setup

### **Prerequisites**

1. **Python 3.12+**
2. **Redis** (for Celery)
3. **PostgreSQL** (via Pixeltable - auto-managed)
4. **API Keys**: Groq + Gemini

---

## 📦 Installation

```bash
# 1. Clone repository
git clone <repo-url>
cd Self_Improving_Prompt_Optimization_API

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment variables
cp .env.example .env
# Edit .env and add:
#   GROQ_API_KEY=your_groq_key
#   GEMINI_API_KEY=your_gemini_key
```

---

## 🔧 Configuration

### **Install Redis**

#### **Windows (via Chocolatey):**

```bash
choco install redis-64
redis-server
```

#### **macOS (via Homebrew):**

```bash
brew install redis
brew services start redis
```

#### **Linux:**

```bash
sudo apt-get install redis-server
sudo systemctl start redis
```

#### **Docker (All platforms):**

```bash
docker run -d -p 6379:6379 redis:latest
```

**Test Redis:**

```bash
redis-cli ping
# Should return: PONG
```

---

## 🚦 Running the System

### **Step 1: Initialize Database**

```bash
python scripts/init_db.py init
```

**Expected output:**

```
✓ Pixeltable home: ./pixeltable_db
✓ Tables: 4 (prompts, datasets, evaluations, optimization_runs)
✓ UDFs: 9
✓ Status: healthy
```

---

### **Step 2: Start the API Server**

```bash
uvicorn app.api.main:app --reload --port 8000
```

**Verify:**

- API Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

---

### **Step 3: Start Celery Worker**

Open a **new terminal** and run:

```bash
# Activate venv first!
source venv/bin/activate  # Windows: venv\Scripts\activate

# Start Celery worker
celery -A app.workers.celery_app worker --loglevel=info
```

**Expected output:**

```
[tasks]
  . app.workers.tasks.run_optimization_job
  . app.workers.tasks.cleanup_old_results
  . app.workers.tasks.health_check
  . app.workers.tasks.batch_optimize

[INFO] celery@hostname ready.
```

---

### **Step 4 (Optional): Start Celery Beat**

For periodic tasks (cleanup):

```bash
celery -A app.workers.celery_app beat --loglevel=info
```

---

## 🧪 Testing

### **Quick Tests**

#### **1. DSPy Integration Test**

```bash
python test_dspy_simple.py
```

**Expected:**

```
✅ DSPy test completed successfully!
- Chain of Thought reasoning working
- Sentiment classification accurate
```

---

#### **2. Production Optimizer Test**

```bash
python test_production_optimizer.py
```

**Expected:**

```
✅ OPTIMIZATION SUCCESSFUL!
Baseline score:    0.667
Optimized score:   0.889
Improvement:       +33.3%
```

---

#### **3. End-to-End API Test**

**Prerequisites:**

- API running (port 8000)
- Redis running
- Celery worker running

```bash
python test_e2e.py
```

**This will:**

1. Upload a dataset via API
2. Create a baseline prompt
3. Start an async optimization job
4. Poll for completion
5. Fetch and display results
6. Test promotion endpoint

**Expected flow:**

```
✓ API is healthy
✓ Dataset uploaded
✓ Prompt created
✓ Job created
[  0%] running: Initializing optimizer...
[ 40%] running: Running Bootstrap optimizer...
[ 70%] running: Evaluating optimized module...
[100%] completed: Optimization complete!

✅ OPTIMIZATION RESULTS
Baseline score:    0.667
Optimized score:   0.889
Improvement:       +33.3%

✅ END-TO-END TEST PASSED!
```

---

## 🐛 Troubleshooting

### **Redis Connection Error**

```
ERROR: Error 111 connecting to localhost:6379. Connection refused.
```

**Fix:**

```bash
# Check if Redis is running
redis-cli ping

# If not, start it:
redis-server  # or: brew services start redis
```

---

### **Celery Task Not Found**

```
ERROR: Received unregistered task of type 'app.workers.tasks.run_optimization_job'
```

**Fix:**

- Restart Celery worker
- Check `app/workers/__init__.py` exists
- Verify task is decorated with `@celery_app.task`

---

### **Pixeltable Connection Error**

```
AssertionError: self._postmaster_info is not None
```

**Fix:**

```bash
# Stop any conflicting Postgres instances
# Then re-initialize
python scripts/init_db.py reset
python scripts/init_db.py init
```

---

### **API Key Errors**

```
ERROR: Invalid API key
```

**Fix:**

```bash
# Check .env file
cat .env | grep API_KEY

# Ensure keys are set:
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=...
```

---

## 📊 Monitoring

### **Celery Flower (UI)**

```bash
pip install flower
celery -A app.workers.celery_app flower --port=5555
```

Visit: http://localhost:5555

**Features:**

- Real-time task monitoring
- Worker status
- Task history
- Performance metrics

---

### **API Metrics**

Access: http://localhost:8000/health

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "pixeltable": {
    "status": "healthy",
    "num_tables": 4
  }
}
```

---

## 🎯 Production Checklist

- [ ] Redis configured and running
- [ ] Celery worker running with `--concurrency=4`
- [ ] Celery Beat running for cleanup
- [ ] API behind reverse proxy (Nginx)
- [ ] SSL/TLS certificates configured
- [ ] Database backups automated
- [ ] Monitoring setup (Sentry, DataDog)
- [ ] Rate limiting enabled
- [ ] Health checks passing

---

## 🔒 Security

### **API Authentication**

All endpoints require API key:

```bash
curl -H "X-API-Key: your_key_here" http://localhost:8000/api/v1/prompts
```

### **Environment Variables**

**Required:**

- `GROQ_API_KEY`
- `GEMINI_API_KEY`

**Optional:**

- `REDIS_HOST` (default: localhost)
- `REDIS_PORT` (default: 6379)
- `REDIS_DB` (default: 0)
- `PIXELTABLE_HOME` (default: ./pixeltable_db)

---

## 📈 Performance Tuning

### **Celery Workers**

```bash
# Run multiple workers for parallel processing
celery -A app.workers.celery_app worker --concurrency=4
```

### **Redis Persistence**

Edit `redis.conf`:

```
save 900 1
save 300 10
appendonly yes
```

### **API Uvicorn**

```bash
# Production mode with Gunicorn
gunicorn app.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

---

## 🎓 Next Steps

1. ✅ Run all tests
2. ✅ Monitor Celery Flower dashboard
3. ✅ Check API docs at `/docs`
4. ✅ Review `VC_PITCH.md` for business context
5. 🚀 Deploy to production!

---

**Need help?** Check logs:

- API: `uvicorn` console output
- Celery: Worker terminal
- Pixeltable: `./pixeltable_db/log`
