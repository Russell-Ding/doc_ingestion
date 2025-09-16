import asyncio
import base64
import mimetypes
from typing import List, Dict, Any, Optional
from pathlib import Path
import structlog
import uuid
from dataclasses import dataclass

from app.core.config import settings
from app.services.bedrock import bedrock_service
from app.services.document_processor import DocumentChunk
from app.services.rag_system import rag_system

logger = structlog.get_logger(__name__)


@dataclass
class SonnetProcessingResult:
    """Result of Sonnet document processing"""
    success: bool
    document_id: str
    extracted_text: str
    chunks: List[DocumentChunk]
    metadata: Dict[str, Any]
    error: Optional[str] = None


class SonnetFallbackProcessor:
    """Ultimate fallback document processor using Claude Sonnet for text extraction"""

    def __init__(self):
        self.max_file_size = 10 * 1024 * 1024  # 10MB limit for Sonnet processing
        # Focus on image formats where Sonnet vision excels
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pdf', '.txt'}

    async def process_document_with_sonnet(
        self,
        file_path: str,
        document_name: str,
        processing_options: Optional[Dict[str, Any]] = None
    ) -> SonnetProcessingResult:
        """Process document using Claude Sonnet as ultimate fallback"""

        try:
            file_path = Path(file_path)

            # Validate file
            if not file_path.exists():
                return SonnetProcessingResult(
                    success=False,
                    document_id="",
                    extracted_text="",
                    chunks=[],
                    metadata={},
                    error=f"File not found: {file_path}"
                )

            # Check file size
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                return SonnetProcessingResult(
                    success=False,
                    document_id="",
                    extracted_text="",
                    chunks=[],
                    metadata={},
                    error=f"File too large for Sonnet processing: {file_size / 1024 / 1024:.1f}MB (max: {self.max_file_size / 1024 / 1024:.1f}MB)"
                )

            # Check file format
            file_extension = file_path.suffix.lower()
            if file_extension not in self.supported_formats:
                return SonnetProcessingResult(
                    success=False,
                    document_id="",
                    extracted_text="",
                    chunks=[],
                    metadata={},
                    error=f"Unsupported file format: {file_extension}"
                )

            logger.info("Starting Sonnet fallback processing", file_path=str(file_path), file_size=file_size)

            # Generate document ID
            document_id = str(uuid.uuid4())

            # Process with Sonnet
            extracted_text = await self._extract_text_with_sonnet(file_path, processing_options)

            if not extracted_text:
                return SonnetProcessingResult(
                    success=False,
                    document_id=document_id,
                    extracted_text="",
                    chunks=[],
                    metadata={},
                    error="Sonnet failed to extract text from document"
                )

            # Create chunks from extracted text
            chunks = await self._create_chunks_from_text(extracted_text, document_id, document_name)

            # Prepare metadata
            metadata = {
                "processing_method": "sonnet_fallback",
                "file_name": file_path.name,
                "file_size": file_size,
                "file_extension": file_extension,
                "chunk_count": len(chunks),
                "extracted_text_length": len(extracted_text),
                "processing_options": processing_options or {}
            }

            logger.info(
                "Sonnet fallback processing completed",
                document_id=document_id,
                chunks_created=len(chunks),
                text_length=len(extracted_text)
            )

            return SonnetProcessingResult(
                success=True,
                document_id=document_id,
                extracted_text=extracted_text,
                chunks=chunks,
                metadata=metadata
            )

        except Exception as e:
            logger.error("Sonnet fallback processing failed", error=str(e), file_path=str(file_path))
            return SonnetProcessingResult(
                success=False,
                document_id="",
                extracted_text="",
                chunks=[],
                metadata={},
                error=f"Sonnet processing failed: {str(e)}"
            )

    async def _extract_text_with_sonnet(
        self,
        file_path: Path,
        processing_options: Optional[Dict[str, Any]] = None
    ) -> str:
        """Extract text using Claude Sonnet"""

        try:
            # Read file content
            with open(file_path, 'rb') as f:
                file_content = f.read()

            # Get MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if not mime_type:
                mime_type = "application/octet-stream"

            # Encode file content for Sonnet
            file_base64 = base64.b64encode(file_content).decode('utf-8')

            # Prepare processing instructions
            processing_mode = processing_options.get("mode", "comprehensive") if processing_options else "comprehensive"
            focus_areas = processing_options.get("focus_areas", []) if processing_options else []

            # Build prompt based on processing options
            prompt = self._build_sonnet_prompt(processing_mode, focus_areas, file_path.suffix.lower())

            # Get file extension
            file_extension = file_path.suffix.lower()

            # Call Sonnet via Bedrock - handle PDFs as images now!
            if mime_type.startswith('image/') or file_extension == '.pdf':
                # For images AND PDFs, use vision capabilities
                if file_extension == '.pdf':
                    # Convert ALL pages of PDF to images and process each
                    try:
                        import fitz  # PyMuPDF
                        pdf_doc = fitz.open(str(file_path))
                        total_pages = len(pdf_doc)

                        logger.info("Processing multi-page PDF", file_path=str(file_path), total_pages=total_pages)

                        # Process all pages (with reasonable limit to avoid overloading)
                        max_pages = min(total_pages, 20)  # Limit to 20 pages for performance
                        if total_pages > max_pages:
                            logger.warning("PDF has many pages, processing first 20 only",
                                         total_pages=total_pages, processing_pages=max_pages)

                        all_extracted_text = []

                        for page_num in range(max_pages):
                            try:
                                logger.info("Processing PDF page", page_num=page_num + 1, total=max_pages)

                                # Convert page to image
                                page = pdf_doc[page_num]
                                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                                img_data = pix.tobytes("png")
                                page_base64 = base64.b64encode(img_data).decode('utf-8')

                                # Create page-specific prompt
                                page_prompt = f"""
{prompt}

This is page {page_num + 1} of {max_pages} from a PDF document ({file_path.name}).
Please extract all text content from this page while maintaining structure and context.

Focus on:
1. Preserving section headers and titles
2. Maintaining paragraph structure
3. Including all numerical data and tables
4. Noting page references when relevant
"""

                                # Process page with Sonnet
                                response = await bedrock_service.analyze_document_image(
                                    image_base64=page_base64,
                                    prompt=page_prompt,
                                    max_tokens=4000
                                )

                                if response and response.get("extracted_text"):
                                    page_text = response["extracted_text"]
                                    # Add page marker
                                    page_text_with_marker = f"\n\n--- PAGE {page_num + 1} ---\n\n{page_text}"
                                    all_extracted_text.append(page_text_with_marker)
                                    logger.info("Successfully processed PDF page",
                                              page_num=page_num + 1,
                                              text_length=len(page_text))
                                else:
                                    logger.warning("Empty response for PDF page", page_num=page_num + 1)
                                    all_extracted_text.append(f"\n\n--- PAGE {page_num + 1} ---\n\n[No text extracted from this page]")

                                # Small delay to avoid rate limiting
                                await asyncio.sleep(0.5)

                            except Exception as page_error:
                                logger.error("Failed to process PDF page",
                                           page_num=page_num + 1,
                                           error=str(page_error))
                                all_extracted_text.append(f"\n\n--- PAGE {page_num + 1} ---\n\n[Error processing this page: {str(page_error)}]")
                                continue

                        pdf_doc.close()

                        # Combine all pages into final extracted text
                        extracted_text = "\n\n".join(all_extracted_text)

                        if total_pages > max_pages:
                            extracted_text += f"\n\n--- NOTE ---\nThis PDF contained {total_pages} pages. Only the first {max_pages} pages were processed due to performance limits."

                        logger.info("Completed multi-page PDF processing",
                                  file_path=str(file_path),
                                  pages_processed=len(all_extracted_text),
                                  total_text_length=len(extracted_text))

                    except Exception as e:
                        logger.error("Failed to convert PDF pages to images", error=str(e))
                        extracted_text = f"❌ Failed to convert PDF to images: {str(e)}"
                        return extracted_text

                else:
                    # Single image processing
                    response = await bedrock_service.analyze_document_image(
                        image_base64=file_base64,
                        prompt=prompt,
                        max_tokens=4000
                    )
                    # Extract the text from the vision response
                    if response and response.get("extracted_text"):
                        extracted_text = response["extracted_text"]
                    else:
                        logger.warning("Sonnet vision returned empty response", file_path=str(file_path))
                        return ""
            elif file_extension == '.txt':
                # For text files, read directly and enhance with AI
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_text = f.read()

                enhance_prompt = f"""
{prompt}

Please clean, structure and enhance the following text content:

{raw_text}

Focus on:
1. Cleaning up any formatting issues
2. Organizing into logical sections
3. Preserving all important information
4. Improving readability while maintaining accuracy
"""

                response = await bedrock_service.generate_text(
                    prompt=enhance_prompt,
                    max_tokens=4000,
                    temperature=0.1
                )

                if response and response.get("content"):
                    extracted_text = response["content"]
                else:
                    # Fallback to raw text if AI processing fails
                    extracted_text = raw_text
            else:
                # For other file types, provide helpful guidance
                extracted_text = f"""
🚨 **Sonnet Fallback Limitation**

The file "{file_path.name}" ({file_extension}) cannot be directly processed by Claude Sonnet via the API.

**Recommended approaches:**
1. **For Word docs**: Export as PDF, then use this tool again
2. **For best results**: Use the regular document upload feature instead

**What Sonnet Fallback works best with:**
- 📷 Images of documents (JPG, PNG, etc.)
- 📄 PDF files (converted to images automatically)
- 📄 Plain text files
- 📋 Screenshots of document pages

**Current file**: {file_extension} format
**MIME type**: {mime_type}

For Word documents, try: Save As → PDF → Upload again with Sonnet fallback!
"""

            # Clean and format extracted text
            extracted_text = self._clean_extracted_text(extracted_text)

            return extracted_text

        except Exception as e:
            logger.error("Error extracting text with Sonnet", error=str(e), file_path=str(file_path))
            return ""

    def _build_sonnet_prompt(self, processing_mode: str, focus_areas: List[str], file_extension: str) -> str:
        """Build appropriate prompt for Sonnet based on processing options"""

        base_prompt = """I need you to extract and format all text content from this document. Please follow these guidelines:

1. Extract ALL readable text, maintaining the logical structure and flow
2. Preserve important formatting like headers, paragraphs, and lists
3. For tables, convert them to a readable text format
4. Include all financial data, numbers, and key metrics
5. Maintain the original meaning and context
6. Remove any artifacts from PDF conversion or OCR errors
7. Structure the output in clear, well-organized sections

"""

        # Add mode-specific instructions
        if processing_mode == "financial":
            base_prompt += """
SPECIAL FOCUS ON FINANCIAL CONTENT:
- Pay special attention to financial statements, balance sheets, income statements
- Preserve all numerical data with proper context
- Include financial ratios, percentages, and calculations
- Extract key financial metrics and performance indicators
- Maintain the relationship between financial line items

"""
        elif processing_mode == "legal":
            base_prompt += """
SPECIAL FOCUS ON LEGAL CONTENT:
- Preserve exact legal language and terminology
- Maintain clause numbering and section references
- Extract all legal obligations, rights, and conditions
- Include definitions and legal interpretations
- Preserve document structure for legal validity

"""
        elif processing_mode == "comprehensive":
            base_prompt += """
COMPREHENSIVE EXTRACTION:
- Extract all content regardless of type
- Include headers, footers, and metadata
- Preserve document structure and hierarchy
- Include any embedded charts or figure descriptions
- Extract both primary content and supplementary information

"""

        # Add focus area instructions
        if focus_areas:
            base_prompt += f"""
ADDITIONAL FOCUS AREAS: {', '.join(focus_areas)}
Pay special attention to content related to these areas.

"""

        # Add file type specific instructions
        if file_extension in ['.pdf']:
            base_prompt += """
PDF SPECIFIC INSTRUCTIONS:
- Handle multi-column layouts appropriately
- Preserve table structures
- Include page numbers or section references if relevant
- Handle any embedded images by describing their content

"""
        elif file_extension in ['.docx', '.doc']:
            base_prompt += """
WORD DOCUMENT INSTRUCTIONS:
- Preserve heading hierarchy
- Include any embedded objects or charts descriptions
- Maintain bullet points and numbering
- Handle track changes if present

"""
        elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            base_prompt += """
IMAGE DOCUMENT INSTRUCTIONS:
- Perform OCR-like extraction of all visible text
- Describe any charts, graphs, or visual elements
- Include table data if present in image
- Preserve text layout and structure

"""

        base_prompt += """
OUTPUT FORMAT:
Provide the extracted text in a clean, well-structured format with clear section breaks.
Use markdown-style headers (# ## ###) to organize content hierarchically.
Separate different sections with clear breaks.
"""

        return base_prompt

    def _clean_extracted_text(self, text: str) -> str:
        """Clean and format extracted text"""

        # Remove excessive whitespace
        import re
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Remove multiple blank lines
        text = re.sub(r'[ \t]+', ' ', text)  # Remove excessive spaces/tabs

        # Remove common artifacts
        artifacts = [
            'base64:',
            'File Type:',
            'File:',
            'Please analyze and extract text from this document',
        ]

        for artifact in artifacts:
            text = text.replace(artifact, '')

        # Clean up the beginning and end
        text = text.strip()

        # Ensure proper paragraph separation
        paragraphs = text.split('\n\n')
        cleaned_paragraphs = [p.strip() for p in paragraphs if p.strip()]

        return '\n\n'.join(cleaned_paragraphs)

    async def _create_chunks_from_text(
        self,
        text: str,
        document_id: str,
        document_name: str
    ) -> List[DocumentChunk]:
        """Create document chunks from extracted text with embeddings"""

        chunks = []

        # Split text into manageable chunks (around 1000-1500 characters)
        chunk_size = 1200
        overlap = 200

        text_parts = text.split('\n\n')  # Split by paragraphs first

        current_chunk = ""
        chunk_index = 0

        for part in text_parts:
            if len(current_chunk) + len(part) + 2 <= chunk_size:  # +2 for \n\n
                if current_chunk:
                    current_chunk += "\n\n" + part
                else:
                    current_chunk = part
            else:
                # Create chunk from current content
                if current_chunk:
                    chunk = DocumentChunk(
                        content=current_chunk.strip(),
                        chunk_index=chunk_index,
                        page_number=None,  # Sonnet extraction doesn't preserve page numbers
                        section_title=self._extract_section_title(current_chunk),
                        chunk_type="text",
                        table_data=None
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                # Start new chunk with overlap if possible
                if len(part) > chunk_size:
                    # Split very long parts
                    words = part.split()
                    current_chunk = " ".join(words[:50])  # Take first 50 words
                else:
                    current_chunk = part

        # Add final chunk
        if current_chunk:
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                chunk_index=chunk_index,
                page_number=None,
                section_title=self._extract_section_title(current_chunk),
                chunk_type="text",
                table_data=None
            )
            chunks.append(chunk)

        # Generate embeddings for all chunks
        logger.info("Generating embeddings for Sonnet chunks", chunk_count=len(chunks))

        if chunks:
            try:
                # Prepare text content for embedding generation
                chunk_texts = [chunk.content for chunk in chunks]

                # Generate embeddings using bedrock service
                embeddings = await bedrock_service.generate_embeddings(chunk_texts)

                # Assign embeddings to chunks
                for i, chunk in enumerate(chunks):
                    if i < len(embeddings):
                        chunk.embeddings = embeddings[i]
                    else:
                        logger.warning("Missing embedding for chunk", chunk_index=i)
                        # Provide empty embedding with correct dimension
                        chunk.embeddings = [0.0] * settings.BEDROCK_EMBEDDING_DIMENSION

                logger.info("Successfully generated embeddings for all chunks", chunk_count=len(chunks))

            except Exception as e:
                logger.error("Failed to generate embeddings for Sonnet chunks", error=str(e))
                # Fallback: provide empty embeddings with correct dimension
                for chunk in chunks:
                    chunk.embeddings = [0.0] * settings.BEDROCK_EMBEDDING_DIMENSION

        return chunks

    def _extract_section_title(self, text: str) -> Optional[str]:
        """Extract section title from text chunk"""

        lines = text.split('\n')
        for line in lines[:3]:  # Check first 3 lines
            line = line.strip()
            # Look for markdown headers or capitalized titles
            if line.startswith('#'):
                return line.replace('#', '').strip()
            elif len(line) < 100 and line.isupper() and len(line) > 5:
                return line
            elif len(line) < 80 and ':' in line and line.index(':') < 50:
                return line.split(':')[0].strip()

        return None

    async def process_and_add_to_rag(
        self,
        file_path: str,
        document_name: str,
        processing_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process document with Sonnet and automatically add to RAG system"""

        try:
            # Process with Sonnet
            result = await self.process_document_with_sonnet(file_path, document_name, processing_options)

            if not result.success:
                return {
                    "success": False,
                    "error": result.error,
                    "document_id": None,
                    "chunks_created": 0
                }

            # Add to RAG system
            file_size = Path(file_path).stat().st_size
            await rag_system.add_document_chunks(
                chunks=result.chunks,
                document_id=result.document_id,
                document_name=document_name,
                file_size=file_size
            )

            logger.info(
                "Document processed with Sonnet and added to RAG",
                document_id=result.document_id,
                document_name=document_name,
                chunks=len(result.chunks)
            )

            return {
                "success": True,
                "error": None,
                "document_id": result.document_id,
                "chunks_created": len(result.chunks),
                "extracted_text_length": len(result.extracted_text),
                "processing_method": "sonnet_fallback",
                "metadata": result.metadata
            }

        except Exception as e:
            logger.error("Failed to process and add document to RAG with Sonnet", error=str(e))
            return {
                "success": False,
                "error": f"Sonnet RAG integration failed: {str(e)}",
                "document_id": None,
                "chunks_created": 0
            }


# Global instance
sonnet_fallback_processor = SonnetFallbackProcessor()