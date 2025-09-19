"""
Enhanced Document Processor with LLM-powered summarization for structured data and email processing.
Supports CSV, Excel with table summarization, and Outlook email processing.
"""

import os
import asyncio
import csv
import email
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import aiofiles
import pandas as pd
from PIL import Image
import pdfplumber
from docx import Document
import structlog
from datetime import datetime
import json
import uuid
import base64
from io import BytesIO, StringIO
import tempfile

from app.core.config import settings
from app.services.bedrock import bedrock_service
from app.services.document_processor import DocumentProcessor, DocumentChunk

logger = structlog.get_logger(__name__)


class EnhancedDocumentChunk(DocumentChunk):
    """Enhanced document chunk with additional metadata for structured data"""

    def __init__(
        self,
        content: str,
        chunk_index: int,
        page_number: Optional[int] = None,
        section_title: Optional[str] = None,
        chunk_type: str = "text",
        bbox: Optional[Dict[str, float]] = None,
        table_data: Optional[Dict[str, Any]] = None,
        summary: Optional[str] = None,
        structured_data: Optional[Dict[str, Any]] = None,
        email_metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(content, chunk_index, page_number, section_title, chunk_type, bbox, table_data)
        self.summary = summary  # LLM-generated summary for vector search
        self.structured_data = structured_data  # Original table/structured data
        self.email_metadata = email_metadata  # Email-specific metadata (sender, date, etc.)


class EnhancedDocumentProcessor(DocumentProcessor):
    """Enhanced document processor with LLM summarization capabilities"""

    def __init__(self):
        super().__init__()
        # Add new file type handlers
        self.file_handlers.update({
            '.csv': self._process_csv_enhanced,
            '.xlsx': self._process_excel_enhanced,
            '.xls': self._process_excel_enhanced,
            '.eml': self._process_email,
            '.msg': self._process_outlook_msg
        })

    async def _generate_table_summary(self, df: pd.DataFrame, context: str = "") -> str:
        """Generate LLM summary of table content for vector search"""
        try:
            # Prepare table analysis prompt
            table_info = f"Table Analysis Context: {context}\n\n"
            table_info += f"Table Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
            table_info += f"Columns: {', '.join(df.columns.astype(str))}\n\n"

            # Add column statistics
            table_info += "Column Statistics:\n"
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    stats = df[col].describe()
                    table_info += f"- {col}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}\n"
                else:
                    unique_count = df[col].nunique()
                    table_info += f"- {col}: {unique_count} unique values, most common: {df[col].value_counts().head(1).index[0] if not df[col].empty else 'N/A'}\n"

            # Add sample data (first and last few rows)
            table_info += f"\nFirst 3 rows:\n{df.head(3).to_string(index=False, max_cols=10)}\n"
            if len(df) > 6:
                table_info += f"\nLast 3 rows:\n{df.tail(3).to_string(index=False, max_cols=10)}\n"

            # Create summarization prompt
            prompt = f"""Analyze this table data and create a comprehensive summary suitable for vector search and retrieval.

{table_info}

Please provide:
1. A concise description of what this table contains
2. Key insights about the data patterns, trends, or relationships
3. Notable values, outliers, or interesting findings
4. What business questions this data could answer
5. Summary of the data distribution and characteristics

Make the summary informative and searchable, focusing on content that would help someone decide if this table is relevant to their query.

Summary:"""

            # Get LLM summary
            summary_response = await bedrock_service.generate_text(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.3
            )

            if summary_response and 'generated_text' in summary_response:
                return summary_response['generated_text'].strip()
            else:
                logger.warning("Failed to generate table summary, using basic description")
                return self._create_basic_table_summary(df, context)

        except Exception as e:
            logger.error("Error generating table summary", error=str(e))
            return self._create_basic_table_summary(df, context)

    def _create_basic_table_summary(self, df: pd.DataFrame, context: str = "") -> str:
        """Create basic table summary without LLM"""
        summary = f"Table: {context}\n"
        summary += f"Contains {df.shape[0]} rows and {df.shape[1]} columns.\n"
        summary += f"Columns: {', '.join(df.columns.astype(str))}\n"

        # Add basic statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary += f"Numeric columns: {', '.join(numeric_cols)}\n"

        text_cols = df.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            summary += f"Text columns: {', '.join(text_cols)}\n"

        return summary

    async def _process_csv_enhanced(
        self,
        file_path: str,
        document_name: str,
        document_id: str
    ) -> List[EnhancedDocumentChunk]:
        """Process CSV files with LLM summarization"""
        chunks = []
        chunk_index = 0

        try:
            # Read CSV with encoding detection
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None

            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"Successfully read CSV with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                raise ValueError("Could not read CSV file with any supported encoding")

            if df.empty:
                logger.warning("CSV file is empty")
                return chunks

            # Generate LLM summary for vector search
            context = f"CSV file: {document_name}"
            table_summary = await self._generate_table_summary(df, context)

            # Create summary chunk for vector search
            summary_chunk = EnhancedDocumentChunk(
                content=table_summary,
                chunk_index=chunk_index,
                section_title=f"CSV Summary: {document_name}",
                chunk_type="table_summary",
                summary=table_summary,
                structured_data={
                    "file_type": "csv",
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "sample_data": df.head(5).to_dict('records')
                }
            )
            chunks.append(summary_chunk)
            chunk_index += 1

            # Create detailed table chunk with full data
            table_content = f"Complete CSV Data: {document_name}\n\n"
            table_content += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n"
            table_content += df.to_string(index=False, max_rows=50)

            if len(df) > 50:
                table_content += f"\n\n... ({len(df) - 50} more rows not shown)"

            detailed_chunk = EnhancedDocumentChunk(
                content=table_content,
                chunk_index=chunk_index,
                section_title=f"CSV Data: {document_name}",
                chunk_type="table_data",
                summary=table_summary,
                structured_data={
                    "file_type": "csv",
                    "full_data": df.to_dict('records'),
                    "columns": list(df.columns),
                    "shape": df.shape
                }
            )
            chunks.append(detailed_chunk)
            chunk_index += 1

            # Create column-specific chunks for detailed analysis
            for col in df.columns:
                col_summary = await self._generate_column_summary(df[col], col, document_name)

                col_chunk = EnhancedDocumentChunk(
                    content=col_summary,
                    chunk_index=chunk_index,
                    section_title=f"Column: {col}",
                    chunk_type="column_analysis",
                    summary=col_summary,
                    structured_data={
                        "column_name": col,
                        "data_type": str(df[col].dtype),
                        "unique_values": df[col].nunique(),
                        "sample_values": df[col].dropna().head(10).tolist()
                    }
                )
                chunks.append(col_chunk)
                chunk_index += 1

            logger.info(f"Processed CSV with {len(chunks)} chunks", file_path=file_path)
            return chunks

        except Exception as e:
            logger.error("CSV processing failed", error=str(e), file_path=file_path)
            raise

    async def _generate_column_summary(self, series: pd.Series, column_name: str, context: str) -> str:
        """Generate summary for individual column"""
        try:
            # Prepare column analysis
            col_info = f"Column Analysis: {column_name} from {context}\n\n"
            col_info += f"Data Type: {series.dtype}\n"
            col_info += f"Total Values: {len(series)}\n"
            col_info += f"Non-null Values: {series.count()}\n"
            col_info += f"Unique Values: {series.nunique()}\n\n"

            if series.dtype in ['int64', 'float64']:
                stats = series.describe()
                col_info += f"Statistics:\n- Min: {stats['min']}\n- Max: {stats['max']}\n- Mean: {stats['mean']:.2f}\n- Std: {stats['std']:.2f}\n\n"

            # Sample values
            col_info += f"Sample Values:\n{series.dropna().head(10).tolist()}\n"

            if series.dtype == 'object':
                value_counts = series.value_counts().head(5)
                col_info += f"\nMost Common Values:\n{value_counts.to_string()}\n"

            # Simple summary without LLM for performance
            summary = f"Column '{column_name}' contains {series.count()} values of type {series.dtype}. "

            if series.dtype in ['int64', 'float64']:
                summary += f"Range from {series.min()} to {series.max()} with mean {series.mean():.2f}."
            else:
                summary += f"Has {series.nunique()} unique values."

            return col_info + f"\nSummary: {summary}"

        except Exception as e:
            logger.error("Error generating column summary", error=str(e), column=column_name)
            return f"Column: {column_name}, Type: {series.dtype}, Values: {len(series)}"

    async def _process_excel_enhanced(
        self,
        file_path: str,
        document_name: str,
        document_id: str
    ) -> List[EnhancedDocumentChunk]:
        """Process Excel files with enhanced LLM summarization"""
        chunks = []
        chunk_index = 0

        try:
            excel_file = pd.ExcelFile(file_path)

            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)

                if not df.empty:
                    # Generate LLM summary for this sheet
                    context = f"Excel sheet '{sheet_name}' from {document_name}"
                    sheet_summary = await self._generate_table_summary(df, context)

                    # Create summary chunk for vector search
                    summary_chunk = EnhancedDocumentChunk(
                        content=sheet_summary,
                        chunk_index=chunk_index,
                        section_title=f"Sheet Summary: {sheet_name}",
                        chunk_type="table_summary",
                        summary=sheet_summary,
                        structured_data={
                            "file_type": "excel",
                            "sheet_name": sheet_name,
                            "shape": df.shape,
                            "columns": list(df.columns),
                            "sample_data": df.head(5).to_dict('records')
                        }
                    )
                    chunks.append(summary_chunk)
                    chunk_index += 1

                    # Create detailed table chunk
                    table_content = f"Excel Sheet: {sheet_name}\n\n"
                    table_content += sheet_summary + "\n\n"
                    table_content += "Full Table Data:\n"
                    table_content += df.to_string(index=False, max_rows=100)

                    if len(df) > 100:
                        table_content += f"\n\n... ({len(df) - 100} more rows available in structured data)"

                    detailed_chunk = EnhancedDocumentChunk(
                        content=table_content,
                        chunk_index=chunk_index,
                        section_title=f"Excel Data: {sheet_name}",
                        chunk_type="table_data",
                        summary=sheet_summary,
                        structured_data={
                            "file_type": "excel",
                            "sheet_name": sheet_name,
                            "full_data": df.to_dict('records'),
                            "columns": list(df.columns),
                            "shape": df.shape
                        }
                    )
                    chunks.append(detailed_chunk)
                    chunk_index += 1

            logger.info(f"Processed Excel file with {len(chunks)} chunks", file_path=file_path)
            return chunks

        except Exception as e:
            logger.error("Excel processing failed", error=str(e), file_path=file_path)
            raise

    async def _process_email(
        self,
        file_path: str,
        document_name: str,
        document_id: str
    ) -> List[EnhancedDocumentChunk]:
        """Process standard email (.eml) files"""
        chunks = []
        chunk_index = 0

        try:
            with open(file_path, 'rb') as f:
                email_message = email.message_from_bytes(f.read())

            # Extract email metadata
            email_data = {
                'from': email_message.get('From', ''),
                'to': email_message.get('To', ''),
                'subject': email_message.get('Subject', ''),
                'date': email_message.get('Date', ''),
                'cc': email_message.get('Cc', ''),
                'bcc': email_message.get('Bcc', '')
            }

            # Extract email body
            body_text = ""
            for part in email_message.walk():
                if part.get_content_type() == "text/plain":
                    body_text += part.get_payload(decode=True).decode('utf-8', errors='ignore')

            # Process email with sender analysis
            email_chunks = await self._process_email_content(
                email_data, body_text, chunk_index, document_name
            )
            chunks.extend(email_chunks)

            logger.info(f"Processed email with {len(chunks)} chunks", file_path=file_path)
            return chunks

        except Exception as e:
            logger.error("Email processing failed", error=str(e), file_path=file_path)
            raise

    async def _process_outlook_msg(
        self,
        file_path: str,
        document_name: str,
        document_id: str
    ) -> List[EnhancedDocumentChunk]:
        """Process Outlook MSG files (requires extract_msg library)"""
        chunks = []

        try:
            # Try to import extract_msg
            try:
                import extract_msg
            except ImportError:
                logger.warning("extract_msg library not installed, treating as binary file")
                return await self._process_as_text_fallback(file_path, document_name, document_id)

            # Extract MSG content
            msg = extract_msg.Message(file_path)

            # Extract email metadata
            email_data = {
                'from': getattr(msg, 'sender', ''),
                'to': getattr(msg, 'to', ''),
                'subject': getattr(msg, 'subject', ''),
                'date': str(getattr(msg, 'date', '')),
                'cc': getattr(msg, 'cc', ''),
                'bcc': getattr(msg, 'bcc', '')
            }

            # Extract body
            body_text = getattr(msg, 'body', '')

            # Process email content
            email_chunks = await self._process_email_content(
                email_data, body_text, 0, document_name
            )
            chunks.extend(email_chunks)

            logger.info(f"Processed Outlook MSG with {len(chunks)} chunks", file_path=file_path)
            return chunks

        except Exception as e:
            logger.error("Outlook MSG processing failed", error=str(e), file_path=file_path)
            # Fallback to basic text processing
            return await self._process_as_text_fallback(file_path, document_name, document_id)

    async def _process_email_content(
        self,
        email_data: Dict[str, str],
        body_text: str,
        start_chunk_index: int,
        document_name: str
    ) -> List[EnhancedDocumentChunk]:
        """Process email content with sender analysis and intelligent chunking"""
        chunks = []
        chunk_index = start_chunk_index

        try:
            # Create email summary chunk
            email_summary = await self._generate_email_summary(email_data, body_text)

            summary_chunk = EnhancedDocumentChunk(
                content=email_summary,
                chunk_index=chunk_index,
                section_title=f"Email Summary: {email_data.get('subject', 'No Subject')}",
                chunk_type="email_summary",
                summary=email_summary,
                email_metadata=email_data
            )
            chunks.append(summary_chunk)
            chunk_index += 1

            # Analyze senders and conversations
            sender_analysis = await self._analyze_email_senders(body_text, email_data)

            # Create sender analysis chunk
            sender_chunk = EnhancedDocumentChunk(
                content=sender_analysis,
                chunk_index=chunk_index,
                section_title="Email Sender Analysis",
                chunk_type="sender_analysis",
                summary=f"Analysis of communication patterns and sender contributions",
                email_metadata={
                    **email_data,
                    "analysis_type": "sender_patterns"
                }
            )
            chunks.append(sender_chunk)
            chunk_index += 1

            # Create conversation chunks
            conversation_chunks = await self._create_conversation_chunks(
                body_text, email_data, chunk_index
            )
            chunks.extend(conversation_chunks)

            return chunks

        except Exception as e:
            logger.error("Error processing email content", error=str(e))
            # Fallback: create basic email chunk
            basic_content = f"Email from: {email_data.get('from', 'Unknown')}\n"
            basic_content += f"Subject: {email_data.get('subject', 'No Subject')}\n"
            basic_content += f"Date: {email_data.get('date', 'Unknown')}\n\n"
            basic_content += body_text

            basic_chunk = EnhancedDocumentChunk(
                content=basic_content,
                chunk_index=chunk_index,
                section_title="Email Content",
                chunk_type="email",
                email_metadata=email_data
            )
            return [basic_chunk]

    async def _generate_email_summary(self, email_data: Dict[str, str], body_text: str) -> str:
        """Generate LLM summary of email content"""
        try:
            # Prepare email for analysis
            email_content = f"Email Analysis\n\n"
            email_content += f"From: {email_data.get('from', 'Unknown')}\n"
            email_content += f"To: {email_data.get('to', 'Unknown')}\n"
            email_content += f"Subject: {email_data.get('subject', 'No Subject')}\n"
            email_content += f"Date: {email_data.get('date', 'Unknown')}\n\n"
            email_content += f"Body Content (first 2000 chars):\n{body_text[:2000]}"

            if len(body_text) > 2000:
                email_content += "\n... (content truncated)"

            prompt = f"""Analyze this email and provide a comprehensive summary suitable for vector search and retrieval.

{email_content}

Please provide:
1. A concise summary of the email's main purpose and content
2. Key topics, decisions, or action items discussed
3. The tone and urgency level of the communication
4. Important dates, deadlines, or commitments mentioned
5. What business context or domain this email relates to

Make the summary informative and searchable, focusing on content that would help someone find this email when searching for related topics.

Summary:"""

            summary_response = await bedrock_service.generate_text(
                prompt=prompt,
                max_tokens=800,
                temperature=0.3
            )

            if summary_response and 'generated_text' in summary_response:
                return summary_response['generated_text'].strip()
            else:
                return self._create_basic_email_summary(email_data, body_text)

        except Exception as e:
            logger.error("Error generating email summary", error=str(e))
            return self._create_basic_email_summary(email_data, body_text)

    def _create_basic_email_summary(self, email_data: Dict[str, str], body_text: str) -> str:
        """Create basic email summary without LLM"""
        summary = f"Email from {email_data.get('from', 'Unknown sender')} "
        summary += f"regarding '{email_data.get('subject', 'No subject')}' "
        summary += f"sent on {email_data.get('date', 'unknown date')}. "

        # Extract first few sentences
        sentences = body_text.split('.')[:3]
        content_preview = '. '.join(sentences).strip()
        if content_preview:
            summary += f"Content preview: {content_preview[:200]}..."

        return summary

    async def _analyze_email_senders(self, body_text: str, email_data: Dict[str, str]) -> str:
        """Analyze email senders and their contributions using LLM"""
        try:
            prompt = f"""Analyze this email content and identify different senders/contributors and their key messages.

Email Metadata:
From: {email_data.get('from', 'Unknown')}
Subject: {email_data.get('subject', 'No Subject')}

Email Content:
{body_text[:3000]}

Please:
1. Identify all senders/contributors mentioned or implied in the email thread
2. Summarize what each person communicated or contributed
3. Note any patterns in communication style or topics
4. Identify key decisions, questions, or action items from each contributor
5. Highlight any important relationships or interactions between senders

Provide a structured analysis that would be useful for understanding who said what and finding specific contributions by person.

Sender Analysis:"""

            analysis_response = await bedrock_service.generate_text(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.3
            )

            if analysis_response and 'generated_text' in analysis_response:
                return analysis_response['generated_text'].strip()
            else:
                return self._create_basic_sender_analysis(body_text, email_data)

        except Exception as e:
            logger.error("Error analyzing email senders", error=str(e))
            return self._create_basic_sender_analysis(body_text, email_data)

    def _create_basic_sender_analysis(self, body_text: str, email_data: Dict[str, str]) -> str:
        """Create basic sender analysis without LLM"""
        analysis = f"Sender Analysis for email from {email_data.get('from', 'Unknown')}\n\n"

        # Look for common email patterns
        if "From:" in body_text or "Sent:" in body_text:
            analysis += "This appears to be a forwarded email or email thread with multiple participants.\n"

        # Count email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        found_emails = re.findall(email_pattern, body_text)
        if found_emails:
            analysis += f"Email addresses found in content: {', '.join(set(found_emails))}\n"

        # Look for quoted sections
        quoted_lines = [line for line in body_text.split('\n') if line.strip().startswith('>')]
        if quoted_lines:
            analysis += f"Contains {len(quoted_lines)} quoted lines from previous messages.\n"

        analysis += f"\nPrimary sender: {email_data.get('from', 'Unknown')}"

        return analysis

    async def _create_conversation_chunks(
        self,
        body_text: str,
        email_data: Dict[str, str],
        start_chunk_index: int
    ) -> List[EnhancedDocumentChunk]:
        """Create conversation chunks from email content"""
        chunks = []
        chunk_index = start_chunk_index

        try:
            # Split email into logical sections
            sections = self._split_email_into_sections(body_text)

            for i, section in enumerate(sections):
                if section.strip():
                    # Generate chunk for this section
                    chunk_content = f"Email Section {i+1}:\n{section}"

                    chunk = EnhancedDocumentChunk(
                        content=chunk_content,
                        chunk_index=chunk_index,
                        section_title=f"Email Conversation Part {i+1}",
                        chunk_type="email_section",
                        summary=f"Part of email conversation: {section[:100]}...",
                        email_metadata={
                            **email_data,
                            "section_index": i,
                            "total_sections": len(sections)
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1

            return chunks

        except Exception as e:
            logger.error("Error creating conversation chunks", error=str(e))
            # Fallback: create single chunk
            chunk = EnhancedDocumentChunk(
                content=body_text,
                chunk_index=chunk_index,
                section_title="Email Content",
                chunk_type="email_content",
                email_metadata=email_data
            )
            return [chunk]

    def _split_email_into_sections(self, body_text: str) -> List[str]:
        """Split email into logical sections based on patterns"""
        sections = []

        # Split by common email delimiters
        delimiters = [
            r'\n\s*From:.*?\n',  # Email headers
            r'\n\s*Sent:.*?\n',  # Sent timestamps
            r'\n\s*Subject:.*?\n',  # Subject lines
            r'\n\s*>.*?\n',  # Quoted lines
            r'\n\s*---+\s*\n',  # Separator lines
            r'\n\s*_{5,}\s*\n',  # Underscore separators
        ]

        current_text = body_text
        for delimiter in delimiters:
            parts = re.split(delimiter, current_text, flags=re.MULTILINE)
            if len(parts) > 1:
                sections.extend([part.strip() for part in parts if part.strip()])
                break

        # If no delimiters found, split by paragraphs
        if not sections:
            paragraphs = body_text.split('\n\n')
            sections = [p.strip() for p in paragraphs if p.strip()]

        # Ensure reasonable chunk sizes
        final_sections = []
        for section in sections:
            if len(section) > 2000:  # Split large sections
                words = section.split()
                for i in range(0, len(words), 300):
                    chunk_words = words[i:i+300]
                    final_sections.append(' '.join(chunk_words))
            else:
                final_sections.append(section)

        return final_sections

    async def _process_as_text_fallback(
        self,
        file_path: str,
        document_name: str,
        document_id: str
    ) -> List[EnhancedDocumentChunk]:
        """Fallback processing for unsupported file types"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            chunk = EnhancedDocumentChunk(
                content=content[:10000],  # Limit content size
                chunk_index=0,
                section_title=f"Text Content: {document_name}",
                chunk_type="text_fallback"
            )

            return [chunk]

        except Exception as e:
            logger.error("Fallback text processing failed", error=str(e))
            return []


# Global enhanced processor instance
enhanced_document_processor = EnhancedDocumentProcessor()