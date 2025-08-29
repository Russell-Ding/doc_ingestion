-- Credit Review Report Generation System Database Schema
-- PostgreSQL with pgvector extension for vector embeddings

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";

-- Document storage and metadata
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
    INDEX idx_documents_type (document_type),
    INDEX idx_documents_status (processing_status),
    INDEX idx_documents_upload_date (upload_date)
);

-- Document chunks for RAG system
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    page_number INTEGER,
    section_title VARCHAR(255),
    chunk_type VARCHAR(50) DEFAULT 'text', -- 'text', 'table', 'image', 'header'
    
    -- Vector embeddings for semantic search
    embeddings VECTOR(1536), -- Assuming OpenAI ada-002 dimensions
    
    -- Bounding box for layout preservation (x1, y1, x2, y2)
    bbox JSONB,
    
    -- Table-specific data for Excel and structured content
    table_data JSONB, -- Stores structured table data
    table_headers JSONB, -- Column headers for tables
    
    -- Text statistics
    word_count INTEGER,
    char_count INTEGER,
    
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT unique_chunk_per_document UNIQUE (document_id, chunk_index),
    INDEX idx_chunks_document (document_id),
    INDEX idx_chunks_page (page_number),
    INDEX idx_chunks_type (chunk_type),
    INDEX idx_chunks_embeddings_cosine ON document_chunks USING ivfflat (embeddings vector_cosine_ops) WITH (lists = 100)
);

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
    
    INDEX idx_reports_status (status),
    INDEX idx_reports_created_by (created_by),
    INDEX idx_reports_created_date (created_date)
);

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
    INDEX idx_segments_report (report_id),
    INDEX idx_segments_status (content_status)
);

-- Document references for each segment
CREATE TABLE segment_document_references (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    segment_id UUID NOT NULL REFERENCES report_segments(id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_ids UUID[], -- Array of relevant chunk IDs
    relevance_score DECIMAL(5,4), -- Similarity score from RAG retrieval
    reference_type VARCHAR(50), -- 'primary', 'supporting', 'contextual'
    
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT unique_segment_document UNIQUE (segment_id, document_id),
    INDEX idx_ref_segment (segment_id),
    INDEX idx_ref_document (document_id)
);

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
    source_chunk_ids UUID[],
    supporting_evidence TEXT,
    
    -- Validation metadata
    validator_agent VARCHAR(100), -- Which AI agent performed validation
    validation_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- User actions
    user_reviewed BOOLEAN DEFAULT FALSE,
    user_action VARCHAR(20), -- 'accepted', 'rejected', 'modified'
    user_notes TEXT,
    
    INDEX idx_validations_segment (segment_id),
    INDEX idx_validations_status (validation_status),
    INDEX idx_validations_type (validation_type)
);

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
    
    INDEX idx_exports_report (report_id),
    INDEX idx_exports_date (export_date)
);

-- User management (basic structure)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'user', -- 'admin', 'manager', 'user'
    is_active BOOLEAN DEFAULT TRUE,
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE
);

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
    
    INDEX idx_jobs_status (status),
    INDEX idx_jobs_type (job_type),
    INDEX idx_jobs_created (created_date)
);

-- Create indexes for performance
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
('export_formats', '["word", "pdf"]', 'Supported export formats');