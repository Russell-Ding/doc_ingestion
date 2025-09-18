import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import aiofiles
import pandas as pd
import pytesseract
from PIL import Image
import pdfplumber
import fitz  # PyMuPDF for better PDF image extraction
from docx import Document
from bs4 import BeautifulSoup
import structlog
from datetime import datetime
import json
import uuid
import base64
from io import BytesIO
import cv2
import numpy as np

from app.core.config import settings
from app.services.bedrock import bedrock_service

logger = structlog.get_logger(__name__)


class DocumentChunk:
    """Represents a document chunk with metadata"""
    
    def __init__(
        self,
        content: str,
        chunk_index: int,
        page_number: Optional[int] = None,
        section_title: Optional[str] = None,
        chunk_type: str = "text",
        bbox: Optional[Dict[str, float]] = None,
        table_data: Optional[Dict[str, Any]] = None
    ):
        self.id = str(uuid.uuid4())
        self.content = content
        self.chunk_index = chunk_index
        self.page_number = page_number
        self.section_title = section_title
        self.chunk_type = chunk_type
        self.bbox = bbox
        self.table_data = table_data
        self.word_count = len(content.split()) if content else 0
        self.char_count = len(content) if content else 0
        self.embeddings = None


class DocumentProcessor:
    """Handles document ingestion and processing"""
    
    def __init__(self):
        self.supported_types = {
            '.pdf': self._process_pdf,
            '.docx': self._process_docx,
            '.xlsx': self._process_excel,
            '.xls': self._process_excel,
            '.jpg': self._process_image,
            '.jpeg': self._process_image,
            '.png': self._process_image,
            '.html': self._process_html,
            '.htm': self._process_html,
            '.txt': self._process_text
        }
    
    async def process_document(
        self,
        file_path: str,
        document_name: str,
        document_id: str
    ) -> List[DocumentChunk]:
        """Process a document and return chunks"""
        
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension not in self.supported_types:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        logger.info(
            "Starting document processing",
            document_id=document_id,
            file_type=file_extension,
            file_path=file_path
        )
        
        try:
            processor = self.supported_types[file_extension]
            chunks = await processor(file_path, document_name, document_id)
            
            # Generate embeddings for all chunks
            await self._generate_embeddings(chunks)
            
            logger.info(
                "Document processing completed",
                document_id=document_id,
                chunk_count=len(chunks)
            )
            
            return chunks
            
        except Exception as e:
            logger.error(
                "Document processing failed",
                document_id=document_id,
                error=str(e)
            )
            raise
    
    async def _process_pdf(
        self,
        file_path: str,
        document_name: str,
        document_id: str
    ) -> List[DocumentChunk]:
        """Process PDF files with advanced fallback for image-based PDFs"""
        chunks = []
        chunk_index = 0
        
        try:
            # First attempt: Standard text extraction with pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text content
                    text = page.extract_text()
                    
                    if text and text.strip():
                        # Split text into smaller chunks
                        text_chunks = self._split_text(text)
                        
                        for text_chunk in text_chunks:
                            chunk = DocumentChunk(
                                content=text_chunk,
                                chunk_index=chunk_index,
                                page_number=page_num,
                                chunk_type="text"
                            )
                            chunks.append(chunk)
                            chunk_index += 1
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if table:
                            table_chunk = await self._process_table(
                                table,
                                chunk_index,
                                page_num,
                                f"Table {table_idx + 1}"
                            )
                            chunks.append(table_chunk)
                            chunk_index += 1
            
            # If no chunks extracted, try fallback methods for image-based PDFs
            if len(chunks) == 0:
                logger.warning(
                    "No text extracted from PDF using standard method, trying fallback approaches",
                    document_id=document_id,
                    file_path=file_path
                )
                
                # Try OCR-based extraction
                ocr_chunks = await self._process_pdf_with_ocr(file_path, document_name, document_id)
                if ocr_chunks:
                    chunks.extend(ocr_chunks)
                    logger.info(f"Extracted {len(ocr_chunks)} chunks using OCR fallback")
                
                # If OCR didn't work or produced poor results, try LLM vision
                if len(chunks) == 0 or await self._should_try_llm_vision(chunks):
                    try:
                        llm_chunks = await self._process_pdf_with_llm_vision(file_path, document_name, document_id)
                        if llm_chunks:
                            # Replace OCR chunks with LLM chunks if LLM produced better results
                            if len(llm_chunks) > len(chunks):
                                chunks = llm_chunks
                                logger.info(f"Using LLM vision results: {len(llm_chunks)} chunks")
                            else:
                                chunks.extend(llm_chunks)
                                logger.info(f"Added {len(llm_chunks)} additional chunks from LLM vision")
                    except Exception as llm_error:
                        logger.warning(f"LLM vision fallback failed: {str(llm_error)}")
                
                # Final fallback: Create placeholder chunk with file info
                if len(chunks) == 0:
                    placeholder_chunk = DocumentChunk(
                        content=f"Document: {document_name}\n\nThis appears to be an image-based PDF that could not be processed automatically. The document may contain scanned images, complex layouts, or protected content that requires manual review.",
                        chunk_index=0,
                        chunk_type="placeholder",
                        page_number=1
                    )
                    chunks.append(placeholder_chunk)
                    logger.warning(f"Created placeholder chunk for unprocessable PDF: {document_name}")
            
            logger.info(
                "PDF processing completed",
                document_id=document_id,
                total_chunks=len(chunks),
                processing_method="standard" if chunk_index > 0 else "fallback"
            )
            
            return chunks
            
        except Exception as e:
            logger.error("PDF processing failed", error=str(e), document_id=document_id)
            raise
    
    async def _process_docx(
        self,
        file_path: str,
        document_name: str,
        document_id: str
    ) -> List[DocumentChunk]:
        """Process Word documents"""
        chunks = []
        chunk_index = 0
        
        try:
            doc = Document(file_path)
            current_section = None
            text_buffer = []
            
            for element in doc.element.body:
                if element.tag.endswith('p'):
                    paragraph = doc.paragraphs[len(chunks) // 2]  # Approximate mapping
                    text = paragraph.text.strip()
                    
                    if text:
                        # Check if this is a heading
                        if paragraph.style.name.startswith('Heading'):
                            # Process accumulated text
                            if text_buffer:
                                combined_text = '\n'.join(text_buffer)
                                text_chunks = self._split_text(combined_text)
                                
                                for text_chunk in text_chunks:
                                    chunk = DocumentChunk(
                                        content=text_chunk,
                                        chunk_index=chunk_index,
                                        section_title=current_section,
                                        chunk_type="text"
                                    )
                                    chunks.append(chunk)
                                    chunk_index += 1
                                
                                text_buffer = []
                            
                            current_section = text
                        else:
                            text_buffer.append(text)
                
                elif element.tag.endswith('tbl'):
                    # Process accumulated text first
                    if text_buffer:
                        combined_text = '\n'.join(text_buffer)
                        text_chunks = self._split_text(combined_text)
                        
                        for text_chunk in text_chunks:
                            chunk = DocumentChunk(
                                content=text_chunk,
                                chunk_index=chunk_index,
                                section_title=current_section,
                                chunk_type="text"
                            )
                            chunks.append(chunk)
                            chunk_index += 1
                        
                        text_buffer = []
                    
                    # Process table
                    table_data = self._extract_docx_table(element)
                    if table_data:
                        table_chunk = await self._process_table(
                            table_data,
                            chunk_index,
                            section_title=current_section
                        )
                        chunks.append(table_chunk)
                        chunk_index += 1
            
            # Process any remaining text
            if text_buffer:
                combined_text = '\n'.join(text_buffer)
                text_chunks = self._split_text(combined_text)
                
                for text_chunk in text_chunks:
                    chunk = DocumentChunk(
                        content=text_chunk,
                        chunk_index=chunk_index,
                        section_title=current_section,
                        chunk_type="text"
                    )
                    chunks.append(chunk)
                    chunk_index += 1
            
            return chunks
            
        except Exception as e:
            logger.error("DOCX processing failed", error=str(e))
            raise
    
    async def _process_excel(
        self,
        file_path: str,
        document_name: str,
        document_id: str
    ) -> List[DocumentChunk]:
        """Process Excel files with advanced table extraction"""
        chunks = []
        chunk_index = 0
        
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                if not df.empty:
                    # Create a summary chunk for the sheet
                    summary_content = f"Sheet: {sheet_name}\n"
                    summary_content += f"Dimensions: {df.shape[0]} rows × {df.shape[1]} columns\n"
                    summary_content += f"Columns: {', '.join(df.columns.astype(str))}\n\n"
                    
                    # Add sample data preview
                    if len(df) > 0:
                        summary_content += "Sample data:\n"
                        summary_content += df.head(3).to_string(index=False)
                    
                    summary_chunk = DocumentChunk(
                        content=summary_content,
                        chunk_index=chunk_index,
                        section_title=f"Sheet: {sheet_name}",
                        chunk_type="table_summary"
                    )
                    chunks.append(summary_chunk)
                    chunk_index += 1
                    
                    # Process table data in chunks
                    table_chunk = await self._process_excel_table(
                        df,
                        chunk_index,
                        sheet_name
                    )
                    chunks.append(table_chunk)
                    chunk_index += 1
                    
                    # Create individual row chunks for detailed analysis
                    row_chunks = await self._create_row_chunks(
                        df,
                        chunk_index,
                        sheet_name
                    )
                    chunks.extend(row_chunks)
                    chunk_index += len(row_chunks)
            
            return chunks
            
        except Exception as e:
            logger.error("Excel processing failed", error=str(e))
            raise
    
    async def _process_image(
        self,
        file_path: str,
        document_name: str,
        document_id: str
    ) -> List[DocumentChunk]:
        """Process images using OCR"""
        chunks = []
        
        try:
            # Open and process image
            image = Image.open(file_path)
            
            # Extract text using OCR
            extracted_text = pytesseract.image_to_string(image)
            
            if extracted_text.strip():
                # Split text into chunks
                text_chunks = self._split_text(extracted_text)
                
                for idx, text_chunk in enumerate(text_chunks):
                    chunk = DocumentChunk(
                        content=text_chunk,
                        chunk_index=idx,
                        chunk_type="ocr_text"
                    )
                    chunks.append(chunk)
            else:
                # Create a placeholder chunk for images without extractable text
                chunk = DocumentChunk(
                    content=f"Image file: {document_name} (no extractable text)",
                    chunk_index=0,
                    chunk_type="image"
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error("Image processing failed", error=str(e))
            raise
    
    async def _process_table(
        self,
        table_data: List[List[str]],
        chunk_index: int,
        page_number: Optional[int] = None,
        section_title: Optional[str] = None
    ) -> DocumentChunk:
        """Process table data into a structured chunk"""
        
        if not table_data or len(table_data) < 2:
            return DocumentChunk(
                content="Empty table",
                chunk_index=chunk_index,
                page_number=page_number,
                section_title=section_title,
                chunk_type="table"
            )
        
        try:
            # Convert table to DataFrame for processing
            df = pd.DataFrame(table_data[1:], columns=table_data[0])
            
            # Create human-readable content
            content = f"Table: {section_title or 'Unnamed'}\n\n"
            content += df.to_string(index=False)
            
            # Store structured data
            table_metadata = {
                "headers": table_data[0],
                "rows": table_data[1:],
                "row_count": len(table_data) - 1,
                "column_count": len(table_data[0]),
                "summary_stats": self._generate_table_stats(df)
            }
        except Exception as e:
            logger.warning(f"Error processing table data: {str(e)}")
            # Fallback to simple text representation
            content = f"Table: {section_title or 'Unnamed'}\n\n"
            for row in table_data:
                content += " | ".join(str(cell) for cell in row) + "\n"
            
            table_metadata = {
                "headers": table_data[0] if table_data else [],
                "rows": table_data[1:] if len(table_data) > 1 else [],
                "row_count": len(table_data) - 1 if table_data else 0,
                "column_count": len(table_data[0]) if table_data else 0,
                "summary_stats": {}
            }
        
        return DocumentChunk(
            content=content,
            chunk_index=chunk_index,
            page_number=page_number,
            section_title=section_title,
            chunk_type="table",
            table_data=table_metadata
        )
    
    async def _process_excel_table(
        self,
        df: pd.DataFrame,
        chunk_index: int,
        sheet_name: str
    ) -> DocumentChunk:
        """Process Excel DataFrame into a table chunk"""
        
        # Create detailed content
        content = f"Excel Table: {sheet_name}\n\n"
        
        # Add basic info
        content += f"Dimensions: {df.shape[0]} rows × {df.shape[1]} columns\n"
        content += f"Columns: {', '.join(df.columns.astype(str))}\n\n"
        
        # Add data summary only if there are numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            content += f"Data Summary:\n{numeric_df.describe().to_string()}\n\n"
        
        # For large datasets, only include a sample to avoid exceeding embedding limits
        MAX_ROWS_FOR_FULL_DATA = settings.MAX_EXCEL_ROWS_FOR_FULL_DATA
        MAX_CONTENT_LENGTH = settings.MAX_EXCEL_CONTENT_LENGTH
        
        if len(df) > MAX_ROWS_FOR_FULL_DATA:
            # Include header + first few rows + last few rows for large datasets
            sample_df = pd.concat([
                df.head(20),  # First 20 rows
                df.tail(10)   # Last 10 rows
            ])
            
            content += f"Data Sample (first 20 and last 10 rows out of {len(df)} total):\n"
            sample_content = sample_df.to_string(index=False)
            
            # If still too long, truncate
            if len(sample_content) > MAX_CONTENT_LENGTH:
                sample_content = sample_content[:MAX_CONTENT_LENGTH] + "...\n[Content truncated due to size]"
            
            content += sample_content
        else:
            # For smaller datasets, include all data but with length check
            full_data = df.to_string(index=False)
            
            if len(full_data) > MAX_CONTENT_LENGTH:
                # Even small datasets can have wide columns, so truncate if needed
                content += f"Data Sample (truncated due to width):\n"
                content += full_data[:MAX_CONTENT_LENGTH] + "...\n[Content truncated due to width]"
            else:
                content += f"Full Data:\n{full_data}"
        
        # Store structured data
        # Convert any datetime objects to strings for JSON serialization
        sample_df = df.head(5).copy()
        for col in sample_df.columns:
            if pd.api.types.is_datetime64_any_dtype(sample_df[col]):
                sample_df[col] = sample_df[col].astype(str)
        
        table_metadata = {
            "sheet_name": sheet_name,
            "headers": [str(h) for h in df.columns.tolist()],  # Ensure headers are strings
            "data_types": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
            "row_count": len(df),
            "column_count": len(df.columns),
            "summary_stats": numeric_df.describe().to_dict() if not numeric_df.empty else {},
            "sample_data": sample_df.to_dict('records')
        }
        
        return DocumentChunk(
            content=content,
            chunk_index=chunk_index,
            section_title=f"Sheet: {sheet_name}",
            chunk_type="excel_table",
            table_data=table_metadata
        )
    
    async def _create_row_chunks(
        self,
        df: pd.DataFrame,
        start_index: int,
        sheet_name: str
    ) -> List[DocumentChunk]:
        """Create individual chunks for significant rows"""
        chunks = []
        
        # Only create row chunks for moderately sized datasets
        # Skip row chunks for very large datasets to avoid overwhelming the system
        if 10 < len(df) < 1000:  # Only for datasets between 10-1000 rows
            # Sample every 10th row or important rows
            sample_indices = range(0, len(df), max(1, len(df) // 10))
            
            for i, row_idx in enumerate(sample_indices):
                if i >= 5:  # Limit to 5 row samples
                    break
                
                row = df.iloc[row_idx]
                content = f"Row {row_idx + 1} from {sheet_name}:\n"
                
                # Limit the length of each row representation
                MAX_ROW_CONTENT_LENGTH = 2000
                current_content = content
                
                for col, value in row.items():
                    addition = f"{col}: {value}\n"
                    if len(current_content + addition) > MAX_ROW_CONTENT_LENGTH:
                        current_content += "... [Row content truncated due to length]\n"
                        break
                    current_content += addition
                
                chunk = DocumentChunk(
                    content=current_content,
                    chunk_index=start_index + i,
                    section_title=f"Sheet: {sheet_name}",
                    chunk_type="table_row"
                )
                chunks.append(chunk)
        elif len(df) >= 1000:
            # For very large datasets, log that row chunks are skipped
            logger.info(f"Skipping row chunk creation for large dataset: {len(df)} rows in {sheet_name}")
        
        return chunks
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into appropriately sized chunks with token limit awareness"""

        # Estimate tokens: roughly 1 token = 3.5-4 characters for English
        # Use conservative estimate to stay well under 8192 token limit
        MAX_CHARS_PER_CHUNK = 6000  # ~1500-1700 tokens, safe margin
        MIN_CHARS_PER_CHUNK = 100   # Minimum viable chunk size

        # If text is very short, return as single chunk
        if len(text) <= MAX_CHARS_PER_CHUNK:
            return [text] if text.strip() else []

        words = text.split()
        chunks = []
        current_chunk = []
        current_chars = 0

        for word in words:
            word_chars = len(word) + 1  # +1 for space

            # Check if adding this word would exceed limit
            if current_chars + word_chars > MAX_CHARS_PER_CHUNK and current_chunk:
                # Try to find a good breaking point
                chunk_text = ' '.join(current_chunk)

                # Try to break at sentence boundaries
                sentences = chunk_text.split('. ')
                if len(sentences) > 1 and len(sentences[0]) > MIN_CHARS_PER_CHUNK:
                    # Keep all but the last sentence
                    chunk_text = '. '.join(sentences[:-1]) + '.'
                    remaining_words = sentences[-1].split() + [word]
                    current_chunk = remaining_words
                    current_chars = sum(len(w) + 1 for w in remaining_words)
                else:
                    # Try to break at paragraph boundaries
                    paragraphs = chunk_text.split('\n\n')
                    if len(paragraphs) > 1 and len(paragraphs[0]) > MIN_CHARS_PER_CHUNK:
                        chunk_text = '\n\n'.join(paragraphs[:-1])
                        remaining_words = paragraphs[-1].split() + [word]
                        current_chunk = remaining_words
                        current_chars = sum(len(w) + 1 for w in remaining_words)
                    else:
                        # Break at current boundary
                        current_chunk = [word]
                        current_chars = word_chars

                # Add the chunk if it has meaningful content
                if chunk_text.strip() and len(chunk_text) > MIN_CHARS_PER_CHUNK:
                    chunks.append(chunk_text.strip())
                elif chunk_text.strip():  # Even small chunks are better than losing content
                    chunks.append(chunk_text.strip())
            else:
                current_chunk.append(word)
                current_chars += word_chars

        # Add remaining text
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())

        # Final safety check: if any chunk is still too long, force split it
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > MAX_CHARS_PER_CHUNK:
                logger.warning(f"Force splitting oversized chunk of {len(chunk)} characters")
                # Force split at character boundary
                while len(chunk) > MAX_CHARS_PER_CHUNK:
                    split_point = MAX_CHARS_PER_CHUNK
                    # Try to split at word boundary
                    last_space = chunk.rfind(' ', 0, split_point)
                    if last_space > split_point * 0.8:  # If we found a space reasonably close
                        split_point = last_space

                    final_chunks.append(chunk[:split_point].strip())
                    chunk = chunk[split_point:].strip()

                if chunk.strip():
                    final_chunks.append(chunk.strip())
            else:
                final_chunks.append(chunk)

        return final_chunks
    
    async def _generate_embeddings(self, chunks: List[DocumentChunk]):
        """Generate embeddings for all chunks with token limit validation"""
        texts = [chunk.content for chunk in chunks]

        # Pre-validate chunk sizes to prevent token limit errors
        validated_texts = []
        for i, text in enumerate(texts):
            # Rough token estimate: 1 token ≈ 3.5-4 characters
            estimated_tokens = len(text) / 3.5

            if estimated_tokens > 8000:  # Conservative limit
                logger.warning(f"Chunk {i} estimated at {estimated_tokens:.0f} tokens, truncating")
                # Truncate to safe size
                safe_char_limit = 7000  # ~2000 tokens
                truncated_text = text[:safe_char_limit] + "\n\n[Content truncated due to token limit]"
                validated_texts.append(truncated_text)
                # Update the chunk content as well
                chunks[i].content = truncated_text
                chunks[i].char_count = len(truncated_text)
                chunks[i].word_count = len(truncated_text.split())
            else:
                validated_texts.append(text)

        try:
            embeddings = await bedrock_service.generate_embeddings(validated_texts)

            for chunk, embedding in zip(chunks, embeddings):
                chunk.embeddings = embedding

        except Exception as e:
            logger.error("Failed to generate embeddings", error=str(e))
            # Set empty embeddings as fallback
            for chunk in chunks:
                chunk.embeddings = [0.0] * settings.BEDROCK_EMBEDDING_DIMENSION
    
    def _generate_table_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for table data"""
        stats = {}
        
        for column in df.columns:
            try:
                col_dtype = str(df[column].dtype)
                col_stats = {"type": col_dtype}
                
                # Check if column is numeric
                if pd.api.types.is_numeric_dtype(df[column]):
                    try:
                        col_stats.update({
                            "mean": float(df[column].mean()) if not df[column].isna().all() else None,
                            "std": float(df[column].std()) if not df[column].isna().all() else None,
                            "min": float(df[column].min()) if not df[column].isna().all() else None,
                            "max": float(df[column].max()) if not df[column].isna().all() else None,
                        })
                    except Exception:
                        # Skip statistics if they can't be computed
                        pass
                
                col_stats["null_count"] = int(df[column].isna().sum())
                col_stats["unique_count"] = int(df[column].nunique())
                
                stats[str(column)] = col_stats
            except Exception as e:
                # Skip this column if there's any issue
                logger.warning(f"Skipping stats for column {column}: {str(e)}")
                stats[str(column)] = {"type": "unknown", "error": str(e)}
        
        return stats
    
    def _extract_docx_table(self, table_element) -> Optional[List[List[str]]]:
        """Extract table data from Word document table element"""
        try:
            # This is a simplified extraction - in practice, you'd need
            # more sophisticated table parsing for Word documents
            rows = []
            for row in table_element.iter():
                if row.tag.endswith('tr'):
                    cells = []
                    for cell in row.iter():
                        if cell.tag.endswith('tc'):
                            cell_text = ''.join(cell.itertext()).strip()
                            cells.append(cell_text)
                    if cells:
                        rows.append(cells)
            
            return rows if rows else None
            
        except Exception as e:
            logger.warning("Failed to extract DOCX table", error=str(e))
            return None
    
    async def _process_pdf_with_ocr(
        self,
        file_path: str,
        document_name: str,
        document_id: str
    ) -> List[DocumentChunk]:
        """Extract text from PDF using OCR (for image-based PDFs)"""
        chunks = []
        chunk_index = 0
        
        try:
            # Use PyMuPDF to extract images from PDF pages
            pdf_document = fitz.open(file_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale for better OCR
                img_data = pix.tobytes("png")
                
                # Process with OCR after image preprocessing
                image = Image.open(BytesIO(img_data))

                # Preprocess image for better OCR results
                processed_image = self._preprocess_image_for_ocr(image)

                # Apply OCR with optimized configuration for word spacing
                custom_config = r'--oem 3 --psm 6'  # Removed character whitelist that can cause spacing issues

                try:
                    extracted_text = pytesseract.image_to_string(
                        processed_image,
                        config=custom_config,
                        lang='eng'
                    ).strip()
                    
                    if extracted_text and len(extracted_text) > 10:  # Only process if meaningful text
                        # Clean up common OCR artifacts
                        cleaned_text = self._clean_ocr_text(extracted_text)
                        
                        if cleaned_text:
                            # Split into chunks
                            text_chunks = self._split_text(cleaned_text)
                            
                            for text_chunk in text_chunks:
                                chunk = DocumentChunk(
                                    content=text_chunk,
                                    chunk_index=chunk_index,
                                    page_number=page_num + 1,
                                    chunk_type="ocr_text"
                                )
                                chunks.append(chunk)
                                chunk_index += 1
                        
                        logger.info(f"OCR extracted {len(extracted_text)} characters from page {page_num + 1}")
                    else:
                        logger.warning(f"OCR produced minimal text for page {page_num + 1}")
                        
                except Exception as ocr_error:
                    logger.warning(f"OCR failed for page {page_num + 1}: {str(ocr_error)}")
                    continue
            
            pdf_document.close()
            
            logger.info(f"OCR processing completed: {len(chunks)} chunks from {document_name}")
            return chunks
            
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}", document_id=document_id)
            return []
    
    async def _process_pdf_with_llm_vision(
        self,
        file_path: str,
        document_name: str,
        document_id: str
    ) -> List[DocumentChunk]:
        """Extract text from PDF using LLM vision capabilities (via Bedrock)"""
        chunks = []
        chunk_index = 0
        
        try:
            # Check if Bedrock vision service is available
            if not hasattr(bedrock_service, 'analyze_document_image'):
                logger.warning("Bedrock vision service not available, skipping LLM vision processing")
                return []
            
            pdf_document = fitz.open(file_path)
            
            # Process only first few pages to avoid excessive API calls
            max_pages = min(5, len(pdf_document))
            
            for page_num in range(max_pages):
                page = pdf_document.load_page(page_num)
                
                # Convert page to high-quality image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")
                
                # Encode image for Bedrock
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                
                try:
                    # Use Bedrock vision to analyze the document image
                    analysis_result = await bedrock_service.analyze_document_image(
                        image_base64=img_base64,
                        prompt=f"Analyze this document page from '{document_name}' and extract all visible text content. Preserve the original structure and formatting. Include tables, headers, and any other textual information."
                    )
                    
                    if analysis_result and analysis_result.get('extracted_text'):
                        extracted_text = analysis_result['extracted_text'].strip()
                        
                        if len(extracted_text) > 20:  # Ensure meaningful content
                            # Split into chunks
                            text_chunks = self._split_text(extracted_text)
                            
                            for text_chunk in text_chunks:
                                chunk = DocumentChunk(
                                    content=text_chunk,
                                    chunk_index=chunk_index,
                                    page_number=page_num + 1,
                                    chunk_type="llm_vision_text"
                                )
                                chunks.append(chunk)
                                chunk_index += 1
                            
                            logger.info(f"LLM vision extracted {len(extracted_text)} characters from page {page_num + 1}")
                        
                except Exception as llm_error:
                    logger.warning(f"LLM vision failed for page {page_num + 1}: {str(llm_error)}")
                    continue
            
            pdf_document.close()
            
            logger.info(f"LLM vision processing completed: {len(chunks)} chunks from {document_name}")
            return chunks
            
        except Exception as e:
            logger.error(f"LLM vision processing failed: {str(e)}", document_id=document_id)
            return []
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean up common OCR artifacts and improve word spacing"""
        if not text:
            return ""

        # First pass: Fix obvious OCR character errors that affect spacing
        text = text.replace('|', 'I')  # Vertical bars often mistaken for I
        text = text.replace('rn', 'm')  # Common OCR confusion
        text = text.replace('vv', 'w')  # Another common confusion
        text = text.replace('ii', 'll')  # Double i often should be ll

        # Fix spacing issues: add spaces around punctuation when missing
        import re
        text = re.sub(r'([a-zA-Z])([.!?,:;])([a-zA-Z])', r'\1\2 \3', text)
        text = re.sub(r'([a-zA-Z])([A-Z])', lambda m: f'{m.group(1)} {m.group(2)}' if m.group(1).islower() else m.group(0), text)

        # Fix concatenated words by adding spaces before capital letters in the middle of "words"
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

        # Add spaces after common punctuation if missing
        text = re.sub(r'([.!?,:;])([a-zA-Z])', r'\1 \2', text)

        # Fix number-letter concatenations
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)

        # Clean up excessive whitespace but preserve line breaks
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            # Clean whitespace within the line
            cleaned_line = ' '.join(line.split())

            if len(cleaned_line) < 2:  # Skip very short lines
                continue

            # Count alphabetic characters to filter out garbage
            if cleaned_line:
                alpha_count = sum(1 for c in cleaned_line if c.isalpha())
                total_valid = sum(1 for c in cleaned_line if c.isalnum() or c.isspace() or c in '.,!?:;()-"\'')

                # Keep lines with reasonable character distribution
                if alpha_count >= 2 and (alpha_count / len(cleaned_line)) >= 0.2 and (total_valid / len(cleaned_line)) >= 0.8:
                    cleaned_lines.append(cleaned_line)

        result = '\n'.join(cleaned_lines)

        # Final cleanup: remove any remaining double spaces
        result = re.sub(r'\s+', ' ', result)
        result = result.replace('\n ', '\n').replace(' \n', '\n')

        return result.strip()
    
    async def _should_try_llm_vision(self, existing_chunks: List[DocumentChunk]) -> bool:
        """Determine if LLM vision should be attempted based on OCR results quality"""
        if not existing_chunks:
            return True
        
        # Calculate quality metrics for existing chunks
        total_chars = sum(len(chunk.content) for chunk in existing_chunks)
        total_words = sum(chunk.word_count for chunk in existing_chunks)
        
        # If we have very little content, try LLM vision
        if total_chars < 100 or total_words < 20:
            return True
        
        # Check for OCR quality indicators
        poor_quality_indicators = 0
        total_content = ' '.join(chunk.content for chunk in existing_chunks)
        
        # Look for common OCR issues
        if '|' in total_content:  # Vertical lines often misrecognized
            poor_quality_indicators += 1
        if total_content.count('rn') > total_words * 0.1:  # Excessive 'rn' vs 'm' confusion
            poor_quality_indicators += 1
        if len([c for c in total_content if c.isalpha()]) / len(total_content) < 0.5:  # Low alphabetic ratio
            poor_quality_indicators += 1
        
        # If multiple quality issues detected, try LLM vision
        return poor_quality_indicators >= 2

    def _preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Preprocess image to improve OCR accuracy and word spacing"""
        try:
            # Convert PIL image to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Convert to grayscale
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

            # Apply noise reduction
            denoised = cv2.medianBlur(gray, 3)

            # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)

            # Apply morphological operations to improve text structure
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            processed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

            # Apply Gaussian blur to smooth edges
            smoothed = cv2.GaussianBlur(processed, (1, 1), 0)

            # Apply adaptive thresholding for better text-background separation
            binary = cv2.adaptiveThreshold(
                smoothed,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )

            # Apply dilation to ensure proper character spacing
            spacing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            spaced = cv2.dilate(binary, spacing_kernel, iterations=1)

            # Convert back to PIL image
            processed_pil = Image.fromarray(spaced)

            # Resize if too small (OCR works better on larger images)
            width, height = processed_pil.size
            if width < 1000 or height < 1000:
                scale_factor = max(1000 / width, 1000 / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                processed_pil = processed_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

            logger.debug("Image preprocessing completed successfully")
            return processed_pil

        except Exception as e:
            logger.warning(f"Image preprocessing failed, using original: {str(e)}")
            return image

    async def _process_html(
        self,
        file_path: str,
        document_name: str,
        document_id: str
    ) -> List[DocumentChunk]:
        """Process HTML files (including SEC EDGAR documents)"""
        chunks = []

        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                html_content = await f.read()

            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove script and style elements
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()

            # Extract text content
            text_content = soup.get_text()

            # Clean up text - remove excessive whitespace
            lines = (line.strip() for line in text_content.splitlines())
            text_content = '\n'.join(line for line in lines if line)

            if not text_content.strip():
                logger.warning("No text content extracted from HTML", file_path=file_path)
                return chunks

            # Split into chunks
            text_chunks = self._split_text(text_content)

            # Create DocumentChunk objects
            for i, chunk_text in enumerate(text_chunks):
                if chunk_text.strip():
                    chunk = DocumentChunk(
                        content=chunk_text,
                        chunk_index=i,
                        chunk_type="text",
                        section_title=f"HTML Section {i+1}"
                    )
                    chunks.append(chunk)

            logger.info(
                "HTML processing completed",
                document_id=document_id,
                chunk_count=len(chunks),
                total_chars=len(text_content)
            )

        except Exception as e:
            logger.error(
                "HTML processing failed",
                document_id=document_id,
                error=str(e)
            )
            raise

        return chunks

    async def _process_text(
        self,
        file_path: str,
        document_name: str,
        document_id: str
    ) -> List[DocumentChunk]:
        """Process plain text files (including SEC EDGAR text filings)"""
        chunks = []

        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                text_content = await f.read()

            if not text_content.strip():
                logger.warning("No text content found in file", file_path=file_path)
                return chunks

            # Clean up text - normalize whitespace but preserve structure
            text_content = text_content.strip()

            # Split into chunks
            text_chunks = self._split_text(text_content)

            # Create DocumentChunk objects
            for i, chunk_text in enumerate(text_chunks):
                if chunk_text.strip():
                    chunk = DocumentChunk(
                        content=chunk_text,
                        chunk_index=i,
                        chunk_type="text",
                        section_title=f"Text Section {i+1}"
                    )
                    chunks.append(chunk)

            logger.info(
                "Text processing completed",
                document_id=document_id,
                chunk_count=len(chunks),
                total_chars=len(text_content)
            )

        except Exception as e:
            logger.error(
                "Text processing failed",
                document_id=document_id,
                error=str(e)
            )
            raise

        return chunks


# Global document processor instance
document_processor = DocumentProcessor()