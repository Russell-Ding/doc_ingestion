# 🚀 Quick Start Guide (SQLite + ChromaDB Version)

This simplified version uses **SQLite + ChromaDB** - no PostgreSQL required! Perfect for getting started quickly.

## ⚡ One-Command Setup

```bash
# Clone and run the setup script
python setup_sqlite.py
```

## 📋 Prerequisites

- **Python 3.9+** 
- **Node.js 16+**
- **AWS Account** with Bedrock access
- ✅ **No database installation needed!** (uses SQLite file)

## 🏃 Manual Setup (if you prefer step-by-step)

### 1. Database Setup
```bash
# No database installation needed!
# SQLite database will be created automatically as credit_reports.db

# If you want to create it manually:
python -c "
import asyncio
import aiosqlite

async def create_db():
    with open('database/sqlite_schema.sql', 'r') as f:
        schema = f.read()
    async with aiosqlite.connect('credit_reports.db') as conn:
        await conn.executescript(schema)
        await conn.commit()

asyncio.run(create_db())
"
```

### 2. Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies (no pgvector!)
pip install -r requirements.txt

# Create environment file
cp .env.example .env
# Edit .env with your AWS credentials
```

### 3. Frontend Setup
```bash
cd frontend
npm install
```

### 4. Start Services

**Terminal 1 - Backend:**
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

**Open:** http://localhost:3000

## 🎯 First Use Walkthrough

### Step 1: Upload Documents
- Click "Upload Documents"  
- Add sample files (PDF financial statements, Excel files)
- Wait for processing to complete

### Step 2: Create Report Segments
- Click "Add Segment"
- Choose from templates:
  - **Financial Summary** - Revenue, profitability, cash flow analysis
  - **Risk Assessment** - Credit risk factors and rating
  - **Cash Flow Analysis** - Cash flow patterns and sustainability

### Step 3: Generate Content
- Click "Generate Report" 
- Watch real-time progress
- Review generated content with AI validation

### Step 4: Export Report
- Click "Export Word" or "Export PDF"
- Review validation comments in the exported document

## 🔧 Key Differences from Full Version

**Simplified Architecture:**
- ✅ **ChromaDB** handles all vector operations (no PostgreSQL extensions)
- ✅ **Standard PostgreSQL** for relational data only
- ✅ **Same AI capabilities** (AWS Bedrock integration)
- ✅ **Same UI/UX** experience
- ✅ **Same validation system** with Word comments

**What's Different:**
- No pgvector dependency
- Embeddings stored in ChromaDB instead of PostgreSQL
- Simpler database schema
- Faster setup process

## 🛠️ Configuration

### Essential Settings (.env file)
```env
# Database (SQLite - No installation needed!)
DATABASE_URL=sqlite+aiosqlite:///./credit_reports.db
DATABASE_PATH=./credit_reports.db

# AWS Credentials (Dynamic Mode)
USE_DYNAMIC_BEDROCK_RUNTIME=true
AWS_REGION=us-east-1

# ChromaDB (Local Storage)
CHROMA_PERSIST_DIRECTORY=./chroma_db
VECTOR_DB_TYPE=chroma
```

### AWS Bedrock Setup
1. Enable Bedrock in AWS Console
2. Request access to Claude models
3. Create IAM user with Bedrock permissions
4. Add credentials to .env file

## 📊 System Components

**Document Processing:**
- Multi-format support (PDF, Word, Excel, Images)
- ChromaDB for vector storage
- AWS Bedrock for embeddings

**Report Generation:**
- AI-powered content generation
- Real-time validation with Word comments
- Professional export formats

**UI Features:**
- Drag-and-drop segment builder
- Template library
- Live preview with validation

## 🔍 Troubleshooting

**Common Issues:**

1. **Database Issues**
   ```bash
   # Check if SQLite database exists
   ls -la credit_reports.db
   
   # Recreate if corrupted
   rm credit_reports.db
   python setup_sqlite.py
   ```

2. **ChromaDB Permission Error**
   ```bash
   # Create directory with proper permissions
   mkdir -p backend/chroma_db
   chmod 755 backend/chroma_db
   ```

3. **AWS Bedrock Access Denied**
   - Verify AWS credentials in .env
   - Check Bedrock service availability in your region
   - Request model access in AWS Console

4. **Node.js Module Errors**
   ```bash
   cd frontend
   rm -rf node_modules package-lock.json
   npm install
   ```

## 🚀 Production Deployment

For production deployment:

1. **Database:** Use managed PostgreSQL (RDS)
2. **Storage:** Configure S3 for documents
3. **Scaling:** Deploy with Docker/Kubernetes
4. **Security:** Enable HTTPS, proper authentication
5. **Monitoring:** Add logging and metrics

## ❓ Need Help?

- **Setup Issues:** Check logs in `backend/logs/`
- **Document Processing:** Verify file formats and sizes
- **AI Generation:** Check AWS Bedrock quotas and permissions
- **UI Problems:** Check browser console for errors

---

**🎉 You're ready to generate professional credit review reports with AI!**