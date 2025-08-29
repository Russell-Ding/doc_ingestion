-- Credit Review System SQLite Schema
-- No external database required - single file database!

-- Document storage and metadata
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    s3_bucket TEXT,
    s3_key TEXT,
    document_type TEXT NOT NULL, -- 'pdf', 'docx', 'xlsx', 'image'
    file_size_bytes INTEGER,
    page_count INTEGER,
    upload_date TEXT DEFAULT (datetime('now')),
    processed_date TEXT,
    processing_status TEXT DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed'
    metadata TEXT, -- JSON as TEXT in SQLite
    created_by TEXT, -- User who uploaded the document
    
    -- ChromaDB integration
    chroma_collection TEXT, -- Which ChromaDB collection this document belongs to
    total_chunks INTEGER DEFAULT 0, -- Total number of chunks in ChromaDB
    
    CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed'))
);

CREATE INDEX idx_documents_type ON documents(document_type);
CREATE INDEX idx_documents_status ON documents(processing_status);
CREATE INDEX idx_documents_upload_date ON documents(upload_date);

-- Document chunks metadata (actual embeddings stored in ChromaDB)
CREATE TABLE document_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    page_number INTEGER,
    section_title TEXT,
    chunk_type TEXT DEFAULT 'text', -- 'text', 'table', 'image', 'header'
    
    -- Bounding box for layout preservation (JSON as TEXT)
    bbox TEXT,
    
    -- Table-specific data for Excel and structured content (JSON as TEXT)
    table_data TEXT, -- Stores structured table data
    table_headers TEXT, -- Column headers for tables
    
    -- Text statistics
    word_count INTEGER,
    char_count INTEGER,
    
    -- ChromaDB references
    chroma_id TEXT, -- ID in ChromaDB collection
    chroma_collection TEXT, -- ChromaDB collection name
    
    created_date TEXT DEFAULT (datetime('now')),
    
    UNIQUE(document_id, chunk_index)
);

CREATE INDEX idx_chunks_document ON document_chunks(document_id);
CREATE INDEX idx_chunks_page ON document_chunks(page_number);
CREATE INDEX idx_chunks_type ON document_chunks(chunk_type);
CREATE INDEX idx_chunks_chroma_id ON document_chunks(chroma_id);

-- Reports table
CREATE TABLE reports (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'draft', -- 'draft', 'generating', 'validating', 'completed', 'error'
    created_by TEXT NOT NULL,
    created_date TEXT DEFAULT (datetime('now')),
    updated_date TEXT DEFAULT (datetime('now')),
    completed_date TEXT,
    
    -- Report configuration (JSON as TEXT)
    template_id TEXT, -- For future template support
    settings TEXT, -- Report-specific settings as JSON
    
    CHECK (status IN ('draft', 'generating', 'validating', 'completed', 'error'))
);

CREATE INDEX idx_reports_status ON reports(status);
CREATE INDEX idx_reports_created_by ON reports(created_by);
CREATE INDEX idx_reports_created_date ON reports(created_date);

-- Report segments (user-defined sections)
CREATE TABLE report_segments (
    id TEXT PRIMARY KEY,
    report_id TEXT NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    prompt TEXT NOT NULL,
    order_index INTEGER NOT NULL,
    
    -- Generated content
    generated_content TEXT,
    content_status TEXT DEFAULT 'pending', -- 'pending', 'generating', 'completed', 'error'
    
    -- Document references (JSON as TEXT)
    required_document_types TEXT, -- JSON array of document types needed
    referenced_documents TEXT, -- JSON array of document IDs used
    
    -- Generation metadata
    generation_date TEXT,
    generation_cost_estimate REAL, -- AWS Bedrock cost tracking
    token_count INTEGER,
    
    created_date TEXT DEFAULT (datetime('now')),
    updated_date TEXT DEFAULT (datetime('now')),
    
    UNIQUE(report_id, order_index),
    CHECK (content_status IN ('pending', 'generating', 'completed', 'error'))
);

CREATE INDEX idx_segments_report ON report_segments(report_id);
CREATE INDEX idx_segments_status ON report_segments(content_status);

-- Document references for each segment
CREATE TABLE segment_document_references (
    id TEXT PRIMARY KEY,
    segment_id TEXT NOT NULL REFERENCES report_segments(id) ON DELETE CASCADE,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_ids TEXT, -- JSON array of ChromaDB chunk IDs
    relevance_score REAL, -- Similarity score from RAG retrieval
    reference_type TEXT, -- 'primary', 'supporting', 'contextual'
    
    created_date TEXT DEFAULT (datetime('now')),
    
    UNIQUE(segment_id, document_id)
);

CREATE INDEX idx_ref_segment ON segment_document_references(segment_id);
CREATE INDEX idx_ref_document ON segment_document_references(document_id);

-- Validation results for generated content
CREATE TABLE content_validations (
    id TEXT PRIMARY KEY,
    segment_id TEXT NOT NULL REFERENCES report_segments(id) ON DELETE CASCADE,
    validation_type TEXT NOT NULL, -- 'accuracy', 'completeness', 'consistency', 'compliance'
    
    -- Text span being validated
    text_start INTEGER NOT NULL,
    text_end INTEGER NOT NULL,
    highlighted_text TEXT NOT NULL,
    
    -- Validation result
    validation_status TEXT NOT NULL, -- 'pass', 'fail', 'warning', 'info'
    validation_message TEXT NOT NULL,
    confidence_score REAL,
    
    -- Source references for validation
    source_document_id TEXT REFERENCES documents(id),
    source_chunk_ids TEXT, -- JSON array of ChromaDB chunk IDs
    supporting_evidence TEXT,
    
    -- Validation metadata
    validator_agent TEXT, -- Which AI agent performed validation
    validation_date TEXT DEFAULT (datetime('now')),
    
    -- User actions
    user_reviewed INTEGER DEFAULT 0, -- Boolean as INTEGER (0/1)
    user_action TEXT, -- 'accepted', 'rejected', 'modified'
    user_notes TEXT,
    
    CHECK (validation_status IN ('pass', 'fail', 'warning', 'info')),
    CHECK (user_action IS NULL OR user_action IN ('accepted', 'rejected', 'modified')),
    CHECK (user_reviewed IN (0, 1))
);

CREATE INDEX idx_validations_segment ON content_validations(segment_id);
CREATE INDEX idx_validations_status ON content_validations(validation_status);
CREATE INDEX idx_validations_type ON content_validations(validation_type);

-- Export history and final reports
CREATE TABLE report_exports (
    id TEXT PRIMARY KEY,
    report_id TEXT NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    export_type TEXT NOT NULL, -- 'word', 'pdf', 'html'
    file_path TEXT,
    s3_bucket TEXT,
    s3_key TEXT,
    
    -- Export configuration
    include_validations INTEGER DEFAULT 1, -- Boolean as INTEGER
    include_references INTEGER DEFAULT 1, -- Boolean as INTEGER
    export_settings TEXT, -- JSON
    
    -- Export metadata
    export_date TEXT DEFAULT (datetime('now')),
    file_size_bytes INTEGER,
    export_duration_ms INTEGER,
    
    CHECK (export_type IN ('word', 'pdf', 'html')),
    CHECK (include_validations IN (0, 1)),
    CHECK (include_references IN (0, 1))
);

CREATE INDEX idx_exports_report ON report_exports(report_id);
CREATE INDEX idx_exports_date ON report_exports(export_date);

-- User management (basic structure)
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT,
    role TEXT DEFAULT 'user', -- 'admin', 'manager', 'user'
    is_active INTEGER DEFAULT 1, -- Boolean as INTEGER
    created_date TEXT DEFAULT (datetime('now')),
    last_login TEXT,
    
    CHECK (role IN ('admin', 'manager', 'user')),
    CHECK (is_active IN (0, 1))
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);

-- System configuration and settings
CREATE TABLE system_settings (
    id TEXT PRIMARY KEY,
    setting_key TEXT UNIQUE NOT NULL,
    setting_value TEXT NOT NULL, -- JSON as TEXT
    description TEXT,
    updated_by TEXT REFERENCES users(id),
    updated_date TEXT DEFAULT (datetime('now'))
);

-- Processing jobs queue
CREATE TABLE processing_jobs (
    id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL, -- 'document_ingestion', 'content_generation', 'validation', 'export'
    entity_id TEXT NOT NULL, -- Document ID, segment ID, or report ID
    status TEXT DEFAULT 'queued', -- 'queued', 'processing', 'completed', 'failed', 'retrying'
    
    -- Job configuration (JSON as TEXT)
    job_data TEXT NOT NULL,
    priority INTEGER DEFAULT 1,
    
    -- Execution details
    started_date TEXT,
    completed_date TEXT,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    created_date TEXT DEFAULT (datetime('now')),
    
    CHECK (job_type IN ('document_ingestion', 'content_generation', 'validation', 'export')),
    CHECK (status IN ('queued', 'processing', 'completed', 'failed', 'retrying'))
);

CREATE INDEX idx_jobs_status ON processing_jobs(status);
CREATE INDEX idx_jobs_type ON processing_jobs(job_type);
CREATE INDEX idx_jobs_created ON processing_jobs(created_date);

-- Insert default system settings
INSERT INTO system_settings (id, setting_key, setting_value, description) VALUES
('1', 'bedrock_model', '"anthropic.claude-3-sonnet-20240229-v1:0"', 'AWS Bedrock model for content generation'),
('2', 'embedding_model', '"amazon.titan-embed-text-v1"', 'AWS Bedrock model for embeddings'),
('3', 'max_chunk_size', '1000', 'Maximum tokens per document chunk'),
('4', 'chunk_overlap', '100', 'Token overlap between chunks'),
('5', 'validation_enabled', 'true', 'Enable automatic content validation'),
('6', 'export_formats', '["word", "pdf"]', 'Supported export formats'),
('7', 'chroma_persist_directory', '"./chroma_db"', 'ChromaDB persistence directory'),
('8', 'vector_db_type', '"chroma"', 'Vector database type');

-- Insert default admin user
INSERT INTO users (id, username, email, full_name, role) VALUES
('admin-user-1', 'admin', 'admin@creditreview.local', 'System Administrator', 'admin');