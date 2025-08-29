-- Simplified Credit Review System Database Schema
-- Standard PostgreSQL (no pgvector extension required)
-- All vector operations handled by ChromaDB

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Document storage and metadata (no embeddings stored in DB)
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    s3_bucket VARCHAR(100),
    s3_key VARCHAR(500),
    document_type VARCHAR(50) NOT NULL, -- 'pdf', 'docx', 'xlsx', 'image'
    file_size_bytes BIGINT,
    page_count INTEGER,
    upload_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_date TIMESTAMP WITH TIME ZONE,
    processing_status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed'
    metadata JSONB, -- Additional document-specific metadata
    created_by UUID, -- User who uploaded the document
    
    -- ChromaDB integration
    chroma_collection VARCHAR(100), -- Which ChromaDB collection this document belongs to
    total_chunks INTEGER DEFAULT 0, -- Total number of chunks in ChromaDB
    
    CONSTRAINT chk_processing_status CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed'))
);

CREATE INDEX idx_documents_type ON documents(document_type);
CREATE INDEX idx_documents_status ON documents(processing_status);
CREATE INDEX idx_documents_upload_date ON documents(upload_date);

-- Document chunks metadata (actual embeddings stored in ChromaDB)
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    page_number INTEGER,
    section_title VARCHAR(255),
    chunk_type VARCHAR(50) DEFAULT 'text', -- 'text', 'table', 'image', 'header'
    
    -- Bounding box for layout preservation (x1, y1, x2, y2)
    bbox JSONB,
    
    -- Table-specific data for Excel and structured content
    table_data JSONB, -- Stores structured table data
    table_headers JSONB, -- Column headers for tables
    
    -- Text statistics
    word_count INTEGER,
    char_count INTEGER,
    
    -- ChromaDB references
    chroma_id VARCHAR(255), -- ID in ChromaDB collection
    chroma_collection VARCHAR(100), -- ChromaDB collection name
    
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT unique_chunk_per_document UNIQUE (document_id, chunk_index)
);

CREATE INDEX idx_chunks_document ON document_chunks(document_id);
CREATE INDEX idx_chunks_page ON document_chunks(page_number);
CREATE INDEX idx_chunks_type ON document_chunks(chunk_type);
CREATE INDEX idx_chunks_chroma_id ON document_chunks(chroma_id);

-- Reports table
CREATE TABLE reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(20) DEFAULT 'draft', -- 'draft', 'generating', 'validating', 'completed', 'error'
    created_by UUID NOT NULL,
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_date TIMESTAMP WITH TIME ZONE,
    
    -- Report configuration
    template_id UUID, -- For future template support
    settings JSONB, -- Report-specific settings
    
    CONSTRAINT chk_report_status CHECK (status IN ('draft', 'generating', 'validating', 'completed', 'error'))
);

CREATE INDEX idx_reports_status ON reports(status);
CREATE INDEX idx_reports_created_by ON reports(created_by);
CREATE INDEX idx_reports_created_date ON reports(created_date);

-- Report segments (user-defined sections)
CREATE TABLE report_segments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    report_id UUID NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    prompt TEXT NOT NULL,
    order_index INTEGER NOT NULL,
    
    -- Generated content
    generated_content TEXT,
    content_status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'generating', 'completed', 'error'
    
    -- Document references
    required_document_types JSONB, -- Array of document types needed
    referenced_documents JSONB, -- Array of document IDs used
    
    -- Generation metadata
    generation_date TIMESTAMP WITH TIME ZONE,
    generation_cost_estimate DECIMAL(10,4), -- AWS Bedrock cost tracking
    token_count INTEGER,
    
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT unique_segment_order UNIQUE (report_id, order_index),
    CONSTRAINT chk_content_status CHECK (content_status IN ('pending', 'generating', 'completed', 'error'))
);

CREATE INDEX idx_segments_report ON report_segments(report_id);
CREATE INDEX idx_segments_status ON report_segments(content_status);

-- Document references for each segment
CREATE TABLE segment_document_references (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    segment_id UUID NOT NULL REFERENCES report_segments(id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_ids TEXT[], -- Array of ChromaDB chunk IDs
    relevance_score DECIMAL(5,4), -- Similarity score from RAG retrieval
    reference_type VARCHAR(50), -- 'primary', 'supporting', 'contextual'
    
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT unique_segment_document UNIQUE (segment_id, document_id)
);

CREATE INDEX idx_ref_segment ON segment_document_references(segment_id);
CREATE INDEX idx_ref_document ON segment_document_references(document_id);

-- Validation results for generated content
CREATE TABLE content_validations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    segment_id UUID NOT NULL REFERENCES report_segments(id) ON DELETE CASCADE,
    validation_type VARCHAR(50) NOT NULL, -- 'accuracy', 'completeness', 'consistency', 'compliance'
    
    -- Text span being validated
    text_start INTEGER NOT NULL,
    text_end INTEGER NOT NULL,
    highlighted_text TEXT NOT NULL,
    
    -- Validation result
    validation_status VARCHAR(20) NOT NULL, -- 'pass', 'fail', 'warning', 'info'
    validation_message TEXT NOT NULL,
    confidence_score DECIMAL(5,4),
    
    -- Source references for validation
    source_document_id UUID REFERENCES documents(id),
    source_chunk_ids TEXT[], -- Array of ChromaDB chunk IDs
    supporting_evidence TEXT,
    
    -- Validation metadata
    validator_agent VARCHAR(100), -- Which AI agent performed validation
    validation_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- User actions
    user_reviewed BOOLEAN DEFAULT FALSE,
    user_action VARCHAR(20), -- 'accepted', 'rejected', 'modified'
    user_notes TEXT,
    
    CONSTRAINT chk_validation_status CHECK (validation_status IN ('pass', 'fail', 'warning', 'info')),
    CONSTRAINT chk_user_action CHECK (user_action IS NULL OR user_action IN ('accepted', 'rejected', 'modified'))
);

CREATE INDEX idx_validations_segment ON content_validations(segment_id);
CREATE INDEX idx_validations_status ON content_validations(validation_status);
CREATE INDEX idx_validations_type ON content_validations(validation_type);

-- Export history and final reports
CREATE TABLE report_exports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    report_id UUID NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    export_type VARCHAR(20) NOT NULL, -- 'word', 'pdf', 'html'
    file_path VARCHAR(500),
    s3_bucket VARCHAR(100),
    s3_key VARCHAR(500),
    
    -- Export configuration
    include_validations BOOLEAN DEFAULT TRUE,
    include_references BOOLEAN DEFAULT TRUE,
    export_settings JSONB,
    
    -- Export metadata
    export_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    file_size_bytes BIGINT,
    export_duration_ms INTEGER,
    
    CONSTRAINT chk_export_type CHECK (export_type IN ('word', 'pdf', 'html'))
);

CREATE INDEX idx_exports_report ON report_exports(report_id);
CREATE INDEX idx_exports_date ON report_exports(export_date);

-- User management (basic structure)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'user', -- 'admin', 'manager', 'user'
    is_active BOOLEAN DEFAULT TRUE,
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT chk_user_role CHECK (role IN ('admin', 'manager', 'user'))
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);

-- System configuration and settings
CREATE TABLE system_settings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    setting_key VARCHAR(100) UNIQUE NOT NULL,
    setting_value JSONB NOT NULL,
    description TEXT,
    updated_by UUID REFERENCES users(id),
    updated_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Processing jobs queue
CREATE TABLE processing_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_type VARCHAR(50) NOT NULL, -- 'document_ingestion', 'content_generation', 'validation', 'export'
    entity_id UUID NOT NULL, -- Document ID, segment ID, or report ID
    status VARCHAR(20) DEFAULT 'queued', -- 'queued', 'processing', 'completed', 'failed', 'retrying'
    
    -- Job configuration
    job_data JSONB NOT NULL,
    priority INTEGER DEFAULT 1,
    
    -- Execution details
    started_date TIMESTAMP WITH TIME ZONE,
    completed_date TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT chk_job_type CHECK (job_type IN ('document_ingestion', 'content_generation', 'validation', 'export')),
    CONSTRAINT chk_job_status CHECK (status IN ('queued', 'processing', 'completed', 'failed', 'retrying'))
);

CREATE INDEX idx_jobs_status ON processing_jobs(status);
CREATE INDEX idx_jobs_type ON processing_jobs(job_type);
CREATE INDEX idx_jobs_created ON processing_jobs(created_date);

-- Create indexes for JSON fields
CREATE INDEX idx_documents_metadata ON documents USING GIN (metadata);
CREATE INDEX idx_chunks_table_data ON document_chunks USING GIN (table_data);
CREATE INDEX idx_segments_settings ON report_segments USING GIN (required_document_types);
CREATE INDEX idx_validations_chunks ON content_validations USING GIN (source_chunk_ids);

-- Functions for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_date()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_date = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_reports_updated_date
    BEFORE UPDATE ON reports
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_date();

CREATE TRIGGER update_segments_updated_date
    BEFORE UPDATE ON report_segments
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_date();

-- Insert default system settings
INSERT INTO system_settings (setting_key, setting_value, description) VALUES
('bedrock_model', '"anthropic.claude-3-sonnet-20240229-v1:0"', 'AWS Bedrock model for content generation'),
('embedding_model', '"amazon.titan-embed-text-v1"', 'AWS Bedrock model for embeddings'),
('max_chunk_size', '1000', 'Maximum tokens per document chunk'),
('chunk_overlap', '100', 'Token overlap between chunks'),
('validation_enabled', 'true', 'Enable automatic content validation'),
('export_formats', '["word", "pdf"]', 'Supported export formats'),
('chroma_persist_directory', '"./chroma_db"', 'ChromaDB persistence directory'),
('vector_db_type', '"chroma"', 'Vector database type (chroma only for now)');