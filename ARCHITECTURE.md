# Credit Review Report Generation System Architecture

## System Overview

The Credit Review Report Generation System is designed to help credit risk managers create comprehensive reports by leveraging document ingestion, RAG (Retrieval-Augmented Generation), and AI-powered content generation with AWS Bedrock.

## Architecture Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend UI   │    │  Document       │    │   RAG System    │
│   - Segment     │────│  Ingestion      │────│   - Vector DB   │
│     Management  │    │  - PDF/Word     │    │   - Table Store │
│   - Prompt      │    │  - Excel        │    │   - Metadata    │
│     Creation    │    │  - OCR          │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         │              ┌─────────────────┐                │
         │              │   Database      │                │
         └──────────────│   - Documents   │────────────────┘
                        │   - Segments    │
                        │   - Reports     │
                        └─────────────────┘
                                 │
         ┌─────────────────┐    │    ┌─────────────────┐
         │  Agent System   │────┼────│  AWS Bedrock    │
         │  - Doc Finder   │    │    │  - Sonnet       │
         │  - Content Gen  │    │    │  - Embeddings   │
         │  - Validator    │    │    │                 │
         └─────────────────┘    │    └─────────────────┘
                                 │
                        ┌─────────────────┐
                        │  Report Gen     │
                        │  - Word Export  │
                        │  - Validation   │
                        │    Comments     │
                        └─────────────────┘
```

## Component Details

### 1. Document Ingestion Pipeline
- **Input Types**: PDF, Word, Excel, Images
- **Processing**: 
  - Text extraction with layout preservation
  - Table extraction (especially for Excel)
  - OCR for scanned documents
  - Metadata extraction (page numbers, document structure)
- **Output**: Structured content with preserved formatting

### 2. RAG System Architecture
- **Vector Database**: Stores document embeddings for semantic search
- **Table Store**: Specialized storage for structured data (Excel tables)
- **Metadata Index**: Document names, page numbers, section headers
- **Hybrid Search**: Combines semantic similarity with metadata filtering

### 3. Frontend UI Components
- **Document Upload Interface**: Multi-file upload with progress tracking
- **Segment Builder**: Drag-and-drop interface for report structure
- **Prompt Editor**: Rich text editor with document reference suggestions
- **Report Preview**: Live preview of generated content
- **Validation Dashboard**: Review and approve AI-generated content

### 4. Agent System
- **Document Finder Agent**: Identifies relevant documents for each segment
- **Content Generator Agent**: Creates report content using retrieved documents
- **Validation Agent**: Cross-references generated content with source materials

### 5. AWS Bedrock Integration
- **Model**: Claude Sonnet for text generation and analysis
- **Embedding Model**: For document vectorization
- **API Management**: Rate limiting, error handling, cost optimization

### 6. Database Schema
```sql
-- Documents table
documents (
    id UUID PRIMARY KEY,
    name VARCHAR,
    file_path VARCHAR,
    document_type VARCHAR,
    upload_date TIMESTAMP,
    metadata JSONB
);

-- Document chunks for RAG
document_chunks (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id),
    content TEXT,
    page_number INTEGER,
    chunk_index INTEGER,
    embeddings VECTOR,
    table_data JSONB
);

-- Report segments
report_segments (
    id UUID PRIMARY KEY,
    report_id UUID,
    name VARCHAR,
    prompt TEXT,
    generated_content TEXT,
    validation_status VARCHAR,
    order_index INTEGER
);
```

## Technology Stack

### Backend
- **Framework**: FastAPI (Python)
- **Database**: PostgreSQL with pgvector extension
- **Vector DB**: ChromaDB or Pinecone
- **Document Processing**: 
  - PyPDF2/pdfplumber for PDFs
  - python-docx for Word documents
  - openpyxl/pandas for Excel
  - pytesseract for OCR
- **LLM Integration**: boto3 for AWS Bedrock

### Frontend
- **Framework**: React with TypeScript
- **UI Library**: Ant Design or Material-UI
- **State Management**: Redux Toolkit
- **File Upload**: react-dropzone
- **Rich Text Editor**: Draft.js or Quill

### Infrastructure
- **Cloud**: AWS
- **Storage**: S3 for document files
- **Compute**: EC2 or ECS
- **Database**: RDS PostgreSQL
- **AI Services**: Bedrock (Claude Sonnet)

## Data Flow

1. **Document Upload**: Files uploaded to S3, metadata stored in DB
2. **Processing**: Documents chunked, embedded, stored in vector DB
3. **Segment Creation**: User defines report segments with prompts
4. **Content Generation**: 
   - Agent finds relevant documents
   - LLM generates content using retrieved context
   - Content stored with references
5. **Validation**: AI validates content against source documents
6. **Export**: Generate Word document with embedded comments

## Security Considerations

- **Document Security**: Encrypted storage, access controls
- **API Security**: Authentication, rate limiting
- **Data Privacy**: No sensitive data sent to external services
- **Audit Trail**: Complete logging of document access and modifications

## Performance Optimization

- **Caching**: Redis for frequent queries
- **Batch Processing**: Async document processing
- **Connection Pooling**: Database connection optimization
- **CDN**: Static asset delivery
- **Load Balancing**: Multi-instance deployment