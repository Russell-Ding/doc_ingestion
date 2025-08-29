# Credit Review Report Generation System

A comprehensive AI-powered toolkit designed to help credit risk managers generate professional credit review reports by automatically analyzing uploaded documents and generating structured, validated content.

## 🏗️ System Architecture

### Core Components

1. **Document Ingestion Pipeline** - Processes PDFs, Word documents, Excel files, and images with advanced table extraction
2. **RAG (Retrieval-Augmented Generation) System** - Optimized vector database system for mixed document types
3. **AI Agent System** - Intelligent agents for document retrieval, content generation, and validation
4. **AWS Bedrock Integration** - Enterprise-grade LLM integration using Claude Sonnet models
5. **Content Validation System** - AI-powered validation with Word document commenting
6. **React UI** - User-friendly interface for segment definition and prompt management
7. **Word Report Generation** - Professional report export with embedded validation comments

### Technology Stack

**Backend:**
- FastAPI (Python) - REST API framework
- PostgreSQL with pgvector - Database with vector search
- ChromaDB - Vector database for RAG
- AWS Bedrock - LLM services (Claude Sonnet)
- python-docx - Word document generation
- pandas, openpyxl - Excel processing
- PyPDF2, pdfplumber - PDF processing

**Frontend:**
- React with TypeScript
- Ant Design - UI components
- React DnD - Drag and drop functionality
- Redux Toolkit - State management

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 16+
- PostgreSQL 14+
- AWS Account with Bedrock access
- Redis (for caching and background tasks)

### Backend Setup

1. **Install Dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Database Setup**
   ```bash
   # Install PostgreSQL with pgvector
   # Run the database schema
   psql -d credit_reports -f database/schema.sql
   ```

3. **Environment Configuration**
   Create `.env` file in backend directory:
   ```env
   # Database
   POSTGRES_SERVER=localhost
   POSTGRES_USER=credit_app
   POSTGRES_PASSWORD=your_password
   POSTGRES_DB=credit_reports
   
   # AWS Configuration
   AWS_REGION=us-east-1
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   
   # Bedrock Models
   BEDROCK_TEXT_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
   BEDROCK_EMBEDDING_MODEL=amazon.titan-embed-text-v1
   
   # Storage
   S3_BUCKET_NAME=your-documents-bucket
   
   # Other settings
   SECRET_KEY=your_secret_key
   DEBUG=false
   ```

4. **Start the Backend**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

### Frontend Setup

1. **Install Dependencies**
   ```bash
   cd frontend
   npm install
   ```

2. **Start Development Server**
   ```bash
   npm start
   ```

## 📋 Features

### Document Processing
- **Multi-format Support**: PDF, Word, Excel, Images
- **Advanced Table Extraction**: Specialized processing for Excel financial data
- **OCR Integration**: Extract text from scanned documents
- **Metadata Preservation**: Page numbers, section headers, document structure

### Report Generation Workflow

1. **Document Upload**: Upload various document types (financial statements, contracts, etc.)
2. **Segment Definition**: Create custom report sections with specific prompts
3. **AI Processing**: Intelligent document retrieval and content generation
4. **Validation**: Automated content validation against source documents
5. **Export**: Generate professional Word documents with validation comments

### Key Features

- **Template Library**: Pre-built prompts for common credit analysis sections
- **Drag-and-Drop Interface**: Intuitive report structure management
- **Real-time Preview**: Live preview of generated content
- **Validation Comments**: Word document integration with issue highlighting
- **Cost Tracking**: Monitor AWS Bedrock usage and costs
- **Audit Trail**: Complete logging of document access and modifications

## 🎯 Usage Examples

### Creating a Financial Summary Section

1. **Add New Segment**
   ```typescript
   const segment = {
     name: "Financial Summary",
     prompt: "Analyze the financial performance based on the provided statements. Include key metrics such as revenue growth, profitability ratios, and cash flow analysis.",
     required_document_types: ["financial_statements", "cash_flow"]
   };
   ```

2. **AI Processing**
   - System automatically finds relevant documents
   - Extracts financial data and metrics
   - Generates comprehensive analysis
   - Validates against source documents

3. **Generated Output**
   ```
   Financial Summary
   
   Based on the financial statements provided, [Company] demonstrates strong financial performance...
   [Content includes specific metrics with validation comments for any questionable data]
   ```

### Document Types Supported

- **Financial Statements**: Balance sheets, income statements, cash flow statements
- **Tax Returns**: Corporate and personal tax documents
- **Bank Statements**: Account statements and transaction histories  
- **Contracts**: Loan agreements, supplier contracts, customer agreements
- **Business Plans**: Strategic plans and financial projections
- **Collateral Documentation**: Asset appraisals and valuations

## 🔧 Configuration

### Bedrock Model Settings

```python
# In app/core/config.py
BEDROCK_TEXT_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"
BEDROCK_EMBEDDING_MODEL = "amazon.titan-embed-text-v1"
BEDROCK_MAX_TOKENS = 4000
BEDROCK_TEMPERATURE = 0.7
```

### RAG System Configuration

```python
# Document chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MAX_RETRIEVED_CHUNKS = 10
SIMILARITY_THRESHOLD = 0.7
```

### Validation Settings

```python
# Content validation
ENABLE_CONTENT_VALIDATION = True
VALIDATION_CONFIDENCE_THRESHOLD = 0.8
```

## 📊 System Components Deep Dive

### 1. Document Processor (`document_processor.py`)
- Handles multiple file formats with specialized extraction
- Preserves document structure and metadata
- Generates embeddings for RAG system
- Optimized table extraction for Excel files

### 2. RAG System (`rag_system.py`)
- Hybrid search combining semantic and keyword approaches
- Separate collections for text and table data
- Numerical query optimization for financial data
- Intelligent result ranking and deduplication

### 3. Agent System (`agents.py`)
- **DocumentFinderAgent**: Locates relevant documents for segments
- **ContentGeneratorAgent**: Generates report content using retrieved context
- **ValidatorAgent**: Validates content against source materials
- **ReportCoordinatorAgent**: Orchestrates the entire process

### 4. Validation System (`validation_system.py`)
- Multi-layered validation (accuracy, completeness, consistency, compliance)
- Word comment integration for user-friendly feedback
- Configurable validation rules and compliance checks
- Quality scoring and issue prioritization

### 5. Word Export (`word_export.py`)
- Professional report formatting
- Embedded validation comments with color coding
- Executive summary and key findings tables
- Source document appendices

## 🔒 Security & Compliance

- **Data Encryption**: All documents encrypted at rest and in transit
- **Access Controls**: Role-based permissions and audit logging
- **Privacy**: No sensitive data sent to external services unnecessarily  
- **Compliance**: Built-in validation rules for credit analysis standards

## 📈 Performance Optimization

- **Async Processing**: Non-blocking document processing and AI operations
- **Connection Pooling**: Optimized database connections
- **Caching**: Redis-based caching for frequent queries
- **Batch Processing**: Efficient handling of multiple documents
- **Rate Limiting**: AWS Bedrock cost and usage controls

## 🚨 Error Handling

- **Graceful Degradation**: System continues operating with partial failures
- **Retry Logic**: Automatic retry for transient failures
- **Comprehensive Logging**: Structured logging for debugging and monitoring
- **User Feedback**: Clear error messages and recovery suggestions

## 🔮 Future Enhancements

- **Multi-language Support**: Process documents in multiple languages
- **Advanced Analytics**: Enhanced financial ratio analysis and benchmarking
- **Integration APIs**: Connect with external credit databases and services
- **Mobile App**: React Native mobile application for on-the-go access
- **Advanced Templates**: Industry-specific report templates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Contact the development team
- Check the documentation and API reference

---

**Built with ❤️ for Credit Risk Managers**

This system represents a comprehensive solution for automated credit review report generation, combining the power of modern AI with practical financial analysis workflows.