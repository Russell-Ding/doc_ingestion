import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import aiofiles
import pandas as pd
import pytesseract
from PIL import Image
import pdfplumber
from docx import Document
import structlog
from datetime import datetime
import json
import uuid

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
            '.png': self._process_image
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
        """Process PDF files with table extraction"""
        chunks = []
        chunk_index = 0
        
        try:
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
            
            return chunks
            
        except Exception as e:
            logger.error("PDF processing failed", error=str(e))
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
        """Split text into appropriately sized chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += 1
            
            if current_size >= settings.CHUNK_SIZE:
                # Look for a good breaking point
                chunk_text = ' '.join(current_chunk)
                
                # Try to break at sentence boundaries
                sentences = chunk_text.split('. ')
                if len(sentences) > 1:
                    # Keep all but the last sentence
                    chunk_text = '. '.join(sentences[:-1]) + '.'
                    remaining_words = sentences[-1].split()
                    current_chunk = remaining_words
                    current_size = len(remaining_words)
                else:
                    # Break at the chunk boundary
                    current_chunk = []
                    current_size = 0
                
                if chunk_text.strip():
                    chunks.append(chunk_text.strip())
        
        # Add remaining text
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
        
        return chunks
    
    async def _generate_embeddings(self, chunks: List[DocumentChunk]):
        """Generate embeddings for all chunks"""
        texts = [chunk.content for chunk in chunks]
        
        try:
            embeddings = await bedrock_service.generate_embeddings(texts)
            
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


# Global document processor instance
document_processor = DocumentProcessor()