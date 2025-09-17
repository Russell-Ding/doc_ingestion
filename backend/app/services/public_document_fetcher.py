import asyncio
import aiohttp
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import structlog
from pathlib import Path
import tempfile
import uuid
import os
from bs4 import BeautifulSoup
import re
import aiofiles
try:
    from sec_edgar_downloader import Downloader
    SEC_EDGAR_AVAILABLE = True
except ImportError:
    SEC_EDGAR_AVAILABLE = False

from app.core.config import settings
from app.services.document_processor import document_processor
from app.services.rag_system import rag_system

logger = structlog.get_logger(__name__)

# Log the SEC downloader availability after logger is defined
if not SEC_EDGAR_AVAILABLE:
    logger.warning("sec-edgar-downloader not available. Install with: pip install sec-edgar-downloader")


class PublicDocumentFetcher:
    """Service for fetching public company documents from regulatory sources"""

    def __init__(self):
        self.session = None
        self.base_headers = {
            'User-Agent': 'zdingaa@gmail.com',  # Required by SEC
            'Accept': '*/*'
        }

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def fetch_public_company_documents(
        self,
        ticker_symbol: str,
        exchange: str,
        quarter: Optional[str] = None,
        year: int = None,
        filing_types: List[str] = None,
        include_exhibits: bool = False,
        auto_process: bool = True
    ) -> Dict[str, Any]:
        """Main method to fetch public company documents"""

        logger.info(
            "Starting public document fetch",
            ticker=ticker_symbol,
            exchange=exchange,
            quarter=quarter,
            year=year,
            filing_types=filing_types
        )

        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            if exchange.upper() == "US":
                return await self._fetch_sec_documents(
                    ticker_symbol, quarter, year, filing_types, include_exhibits, auto_process
                )
            elif exchange.upper() == "UK":
                return await self._fetch_uk_documents(
                    ticker_symbol, quarter, year, filing_types, include_exhibits, auto_process
                )
            elif exchange.upper() in ["EU", "CANADA"]:
                return await self._fetch_international_documents(
                    ticker_symbol, exchange, quarter, year, filing_types, include_exhibits, auto_process
                )
            else:
                return {
                    "success": False,
                    "error": f"Unsupported exchange: {exchange}",
                    "documents": []
                }

        except Exception as e:
            logger.error("Failed to fetch public company documents", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "documents": []
            }

    async def _fetch_sec_documents(
        self,
        ticker_symbol: str,
        quarter: Optional[str],
        year: int,
        filing_types: List[str],
        include_exhibits: bool,
        auto_process: bool
    ) -> Dict[str, Any]:
        """Fetch SEC EDGAR documents using sec-edgar-downloader"""

        try:
            if not SEC_EDGAR_AVAILABLE:
                return {
                    "success": False,
                    "error": "sec-edgar-downloader package not installed. Please install with: pip install sec-edgar-downloader",
                    "documents": []
                }

            logger.info(
                "Starting SEC document fetch with sec-edgar-downloader",
                ticker=ticker_symbol,
                filing_types=filing_types,
                year=year,
                quarter=quarter
            )

            # Initialize the downloader with proper user agent
            downloader = Downloader("Document Ingestion System", "admin@company.com")

            processed_documents = []

            # Download each filing type
            for filing_type in filing_types:
                try:
                    logger.info(f"Downloading {filing_type} filings for {ticker_symbol}")

                    # Set date filter for the year
                    after_date = f"{year}-01-01"
                    before_date = f"{year}-12-31"

                    # Download filings with date filtering
                    download_result = downloader.get(
                        filing_type,
                        ticker_symbol,
                        after=after_date,
                        before=before_date,
                        limit=10  # Limit to recent filings
                    )

                    logger.info(f"Download result: {download_result} for {filing_type} filings")

                    # The downloader.get() method returns the number of downloaded files, not the files themselves
                    # Files are saved to disk in the default directory structure
                    if download_result and download_result > 0:
                        # Process downloaded files from the directory structure
                        await self._process_downloaded_filings(
                            download_result, ticker_symbol, filing_type, year, quarter, auto_process, processed_documents
                        )

                except Exception as e:
                    logger.warning(f"Failed to download {filing_type} filings", error=str(e))
                    continue

            if not processed_documents:
                return {
                    "success": False,
                    "error": f"No {filing_types} filings found for {ticker_symbol} in {year}" + (f" {quarter}" if quarter else ""),
                    "documents": []
                }

            return {
                "success": True,
                "error": None,
                "documents": processed_documents,
                "company_info": {
                    "ticker": ticker_symbol,
                    "exchange": "US"
                }
            }

        except Exception as e:
            logger.error("SEC document fetch failed", error=str(e))
            return {
                "success": False,
                "error": f"SEC document fetch failed: {str(e)}",
                "documents": []
            }

    def _matches_date_criteria(self, file_path: str, year: int, quarter: Optional[str]) -> bool:
        """Check if downloaded file matches the date criteria"""
        try:
            # Extract date from file path - sec-edgar-downloader uses date-based folder structure
            path_obj = Path(file_path)

            # The file path typically contains date information in folder structure
            # Look for year in the path components
            path_parts = path_obj.parts

            # Simple approach: check if the year appears in the path
            year_str = str(year)
            for part in path_parts:
                if year_str in part:
                    # If quarter is specified, we could add more sophisticated matching
                    # For now, just match by year
                    return True

            # Fallback: always return True to include all files if we can't determine date
            return True

        except Exception as e:
            logger.warning("Error checking date criteria", file_path=file_path, error=str(e))
            return True

    async def _process_downloaded_filings(
        self,
        download_count: int,
        ticker_symbol: str,
        filing_type: str,
        year: int,
        quarter: Optional[str],
        auto_process: bool,
        processed_documents: List[Dict[str, Any]]
    ):
        """Process downloaded filings from sec-edgar-downloader"""
        try:
            # sec-edgar-downloader saves files in a specific directory structure
            # The files are typically saved in ./sec-edgar-filings/[ticker]/[filing_type]/
            base_path = Path("./sec-edgar-filings") / ticker_symbol.upper() / filing_type

            logger.info(f"Looking for downloaded files in: {base_path}")
            logger.info(f"Directory exists: {base_path.exists()}")

            if base_path.exists():
                # List all files in the directory for debugging
                all_files = list(base_path.glob("**/*"))
                logger.info(f"Found {len(all_files)} total files in directory")
                for file_path in all_files[:5]:  # Log first 5 files
                    logger.info(f"File found: {file_path}")

            if base_path.exists():
                # Find all downloaded files
                for file_path in base_path.glob("**/*.txt"):
                    if self._matches_date_criteria(str(file_path), year, quarter):
                        # Clean the file before processing if it's XML/XBRL
                        cleaned_file_path = await self._clean_edgar_file(str(file_path))
                        doc_result = await self._process_downloaded_file(
                            cleaned_file_path, ticker_symbol, filing_type, auto_process
                        )
                        if doc_result:
                            processed_documents.append(doc_result)
                            logger.info("Processed SEC filing",
                                      file_path=str(file_path),
                                      cleaned_path=cleaned_file_path,
                                      filing_type=filing_type)

                # Also check for HTML files
                for file_path in base_path.glob("**/*.htm*"):
                    if self._matches_date_criteria(str(file_path), year, quarter):
                        # Clean the file before processing if it's XML/XBRL
                        cleaned_file_path = await self._clean_edgar_file(str(file_path))
                        doc_result = await self._process_downloaded_file(
                            cleaned_file_path, ticker_symbol, filing_type, auto_process
                        )
                        if doc_result:
                            processed_documents.append(doc_result)
                            logger.info("Processed SEC filing",
                                      file_path=str(file_path),
                                      cleaned_path=cleaned_file_path,
                                      filing_type=filing_type)
            else:
                logger.warning("Download directory not found", path=str(base_path))

        except Exception as e:
            logger.error("Error processing downloaded filings", error=str(e))

    async def _clean_edgar_file(self, file_path: str) -> str:
        """Clean EDGAR XML/XBRL files to extract readable content"""
        try:
            path_obj = Path(file_path)

            # Read the original file
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = await f.read()

            # Check if it's XML/XBRL content
            if self._is_xml_content(content):
                logger.info(f"Cleaning XML/XBRL content from {file_path}")
                cleaned_content = self._extract_text_from_xbrl(content)
            else:
                logger.info(f"File appears to be plain text, minimal cleaning: {file_path}")
                cleaned_content = self._clean_plain_text(content)

            # Create cleaned file
            cleaned_file_path = str(path_obj.parent / f"{path_obj.stem}_cleaned.txt")
            async with aiofiles.open(cleaned_file_path, 'w', encoding='utf-8') as f:
                await f.write(cleaned_content)

            logger.info(f"Cleaned file saved: {cleaned_file_path}")
            return cleaned_file_path

        except Exception as e:
            logger.warning(f"Failed to clean EDGAR file {file_path}, using original: {str(e)}")
            return file_path

    def _is_xml_content(self, content: str) -> bool:
        """Check if content is XML/XBRL format"""
        content_start = content[:1000].lower()
        return any(marker in content_start for marker in [
            '<?xml', '<xbrl', '<html', 'xmlns:', '<sec-document'
        ])

    def _extract_text_from_xbrl(self, content: str) -> str:
        """Extract readable text from XBRL/XML content"""
        try:
            # Use BeautifulSoup to parse XML/HTML
            soup = BeautifulSoup(content, 'html.parser')

            # Remove script, style, and metadata elements
            for element in soup(['script', 'style', 'meta', 'link', 'title']):
                element.decompose()

            # Remove XBRL-specific elements that don't contain useful text
            xbrl_noise_tags = [
                'xbrl', 'context', 'unit', 'schemaref', 'linkbaseref',
                'roleref', 'arcroleref', 'footnotelink', 'loc', 'footnote'
            ]
            for tag in xbrl_noise_tags:
                for element in soup.find_all(tag):
                    element.decompose()

            # Extract text content
            text = soup.get_text()

            # Clean up the text
            return self._clean_extracted_text(text)

        except Exception as e:
            logger.warning(f"Failed to parse XML/XBRL, trying simple text extraction: {str(e)}")
            # Fallback: simple regex-based cleaning
            return self._simple_xml_clean(content)

    def _simple_xml_clean(self, content: str) -> str:
        """Simple regex-based XML cleaning as fallback"""
        import re

        # Remove XML tags
        content = re.sub(r'<[^>]+>', ' ', content)

        # Remove XML declarations and processing instructions
        content = re.sub(r'<\?[^>]*\?>', '', content)

        # Remove CDATA sections
        content = re.sub(r'<!\[CDATA\[.*?\]\]>', '', content, flags=re.DOTALL)

        # Clean up the text
        return self._clean_extracted_text(content)

    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        import re

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove excessive blank lines
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)

        # Remove very short lines (likely noise)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 3:  # Keep lines with more than 3 characters
                cleaned_lines.append(line)

        # Remove duplicate consecutive lines
        final_lines = []
        prev_line = None
        for line in cleaned_lines:
            if line != prev_line:
                final_lines.append(line)
                prev_line = line

        return '\n'.join(final_lines).strip()

    def _clean_plain_text(self, content: str) -> str:
        """Light cleaning for plain text files"""
        # Just normalize whitespace and remove excessive blank lines
        import re
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        return content.strip()

    async def _process_downloaded_file(
        self,
        file_path: str,
        ticker_symbol: str,
        filing_type: str,
        auto_process: bool
    ) -> Optional[Dict[str, Any]]:
        """Process a downloaded SEC filing file"""
        try:
            path_obj = Path(file_path)

            # Get file info
            file_size = path_obj.stat().st_size

            document_info = {
                "filing_type": filing_type,
                "title": f"{ticker_symbol} {filing_type} Filing",
                "date": datetime.now().isoformat(),  # We'll extract actual date from filename if needed
                "url": f"file://{file_path}",
                "file_path": str(file_path),
                "file_size": file_size
            }

            # Auto-process for RAG if requested
            if auto_process:
                try:
                    # Generate unique document ID
                    document_id = str(uuid.uuid4())
                    document_name = f"{ticker_symbol}_{filing_type}_{path_obj.stem}"

                    # Process document
                    chunks = await document_processor.process_document(
                        file_path=str(file_path),
                        document_name=document_name,
                        document_id=document_id
                    )

                    # Add to RAG system
                    await rag_system.add_document_chunks(
                        chunks=chunks,
                        document_id=document_id,
                        document_name=document_name,
                        file_size=file_size
                    )

                    document_info.update({
                        "document_id": document_id,
                        "processed": True,
                        "chunks_created": len(chunks)
                    })

                    logger.info("SEC document processed for RAG",
                              document_id=document_id,
                              chunks=len(chunks))

                except Exception as e:
                    logger.error("Failed to process SEC document for RAG", error=str(e))
                    document_info.update({
                        "processed": False,
                        "error": str(e)
                    })

            return document_info

        except Exception as e:
            logger.error("Error processing downloaded file", file_path=file_path, error=str(e))
            return None

    async def _get_cik_from_ticker(self, ticker_symbol: str) -> Optional[str]:
        """Get SEC CIK number from ticker symbol"""
        try:
            # SEC Company Tickers JSON endpoint
            url = "https://www.sec.gov/files/company_tickers.json"
            headers = {**self.base_headers, 'Host': 'www.sec.gov'}

            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()

                    # Search for ticker in the data
                    for key, company in data.items():
                        if company.get("ticker", "").upper() == ticker_symbol.upper():
                            cik = str(company.get("cik_str", "")).zfill(10)
                            logger.info("Found CIK for ticker", ticker=ticker_symbol, cik=cik)
                            return cik

                    logger.warning("Ticker not found in SEC database", ticker=ticker_symbol)
                    return None
                else:
                    logger.error("Failed to fetch SEC ticker data", status=response.status)
                    return None

        except Exception as e:
            logger.error("Error getting CIK from ticker", error=str(e))
            return None

    async def _search_sec_filings(
        self,
        cik: str,
        filing_types: List[str],
        year: int,
        quarter: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Search for SEC filings"""
        try:
            # SEC EDGAR API endpoint - CIK needs to be padded to 10 digits
            padded_cik = cik.zfill(10)
            url = f"https://data.sec.gov/submissions/CIK{padded_cik}.json"

            logger.info("Searching SEC filings", url=url, cik=cik, padded_cik=padded_cik)

            # Add delay to respect SEC rate limits
            await asyncio.sleep(0.1)
            headers = {**self.base_headers, 'Host': 'data.sec.gov'}

            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    filings = data.get("filings", {}).get("recent", {})

                    # Debug: Log what we got from SEC
                    forms = filings.get("form", [])
                    filing_dates = filings.get("filingDate", [])
                    logger.info("SEC response received",
                              total_forms=len(forms),
                              recent_forms=forms[:10] if forms else [],
                              recent_dates=filing_dates[:10] if filing_dates else [])

                    # Filter filings by type and date
                    filtered_filings = []
                    accession_numbers = filings.get("accessionNumber", [])
                    primary_documents = filings.get("primaryDocument", [])

                    for i, form in enumerate(forms):
                        if form in filing_types:
                            filing_date = filing_dates[i] if i < len(filing_dates) else None
                            if filing_date and str(year) in filing_date:
                                logger.info("Found matching filing",
                                          form=form,
                                          filing_date=filing_date,
                                          quarter_requested=quarter)

                                # TODO: Re-enable quarter filtering after debugging
                                # Additional quarter filtering for 10-Q
                                # if quarter and form == "10-Q":
                                #     # This is a simplified quarter matching - you might want to enhance this
                                #     filing_month = int(filing_date.split("-")[1])
                                #     quarter_months = {
                                #         "Q1": [3, 4, 5],
                                #         "Q2": [6, 7, 8],
                                #         "Q3": [9, 10, 11],
                                #         "Q4": [12, 1, 2]
                                #     }
                                #     if filing_month not in quarter_months.get(quarter, []):
                                #         continue

                                filtered_filings.append({
                                    "form": form,
                                    "filing_date": filing_date,
                                    "accession_number": accession_numbers[i] if i < len(accession_numbers) else None,
                                    "primary_document": primary_documents[i] if i < len(primary_documents) else None
                                })

                    logger.info("Found SEC filings", count=len(filtered_filings), cik=cik)
                    return filtered_filings[:10]  # Return top 10 most recent

                else:
                    response_text = await response.text()
                    logger.error("Failed to search SEC filings",
                               status=response.status,
                               url=url,
                               response_preview=response_text[:500] if response_text else "No content")
                    return []

        except Exception as e:
            logger.error("Error searching SEC filings", error=str(e))
            return []

    async def _download_and_process_sec_document(
        self,
        filing: Dict[str, Any],
        ticker_symbol: str,
        auto_process: bool,
        cik: str
    ) -> Optional[Dict[str, Any]]:
        """Download and optionally process SEC document"""
        try:
            accession_number = filing["accession_number"].replace("-", "")
            primary_document = filing["primary_document"]

            # SEC document URL - try different URL patterns
            cik_for_url = str(int(cik))  # Remove leading zeros for URL path

            # Use SEC EDGAR data API for modern document access
            doc_api_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"

            # Try multiple URL approaches for document download
            # First, try to get the filing details to find the actual document URLs
            filing_detail_url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"

            # New approach: Get the filing index to find all available documents
            base_url = f"https://www.sec.gov/Archives/edgar/data/{cik_for_url}/{filing['accession_number']}"

            # First, try to get the filing index
            index_url = f"{base_url}/index.json"
            headers = {**self.base_headers, 'Host': 'www.sec.gov'}

            logger.info("Getting filing index", url=index_url)

            try:
                async with self.session.get(index_url, headers=headers) as index_response:
                    if index_response.status == 200:
                        index_data = await index_response.json()
                        directory = index_data.get('directory', {})
                        items = directory.get('item', [])

                        # Find documents in the filing
                        available_docs = []
                        for item in items:
                            name = item.get('name', '')
                            if name.endswith(('.htm', '.html', '.txt')) and not name.startswith('R'):
                                available_docs.append(name)

                        logger.info("Found documents in filing", documents=available_docs)

                        # Prioritize document types: txt first (easier to process), then htm
                        url_attempts = []
                        for doc in available_docs:
                            if doc.endswith('.txt'):
                                url_attempts.insert(0, f"{base_url}/{doc}")  # Prioritize txt
                            else:
                                url_attempts.append(f"{base_url}/{doc}")
                    else:
                        logger.warning("Could not get filing index", status=index_response.status)
                        # Fallback to original approach
                        url_attempts = [f"{base_url}/{primary_document}"]

            except Exception as e:
                logger.warning("Error getting filing index", error=str(e))
                # Fallback to original approach
                url_attempts = [f"{base_url}/{primary_document}"]

            response = None
            successful_url = None

            for i, doc_url in enumerate(url_attempts):
                try:
                    # Adjust headers based on URL
                    if 'data.sec.gov' in doc_url:
                        headers = {**self.base_headers, 'Host': 'data.sec.gov'}
                    else:
                        headers = {**self.base_headers, 'Host': 'www.sec.gov'}

                    logger.info(f"Trying URL approach {i+1}", url=doc_url)

                    async with self.session.get(doc_url, headers=headers) as doc_response:
                        if doc_response.status == 200:
                            try:
                                # Read content while response is open
                                if '/ix?' in doc_url:
                                    # For ix viewer, get text content (HTML)
                                    content = await doc_response.text()
                                    content = content.encode('utf-8')  # Convert to bytes for consistency
                                    logger.info("Downloaded ix viewer content", url=doc_url, size=len(content))
                                else:
                                    # For direct documents, get binary content
                                    content = await doc_response.read()
                                    logger.info("Downloaded document content", url=doc_url, size=len(content))

                                successful_url = doc_url
                                response = doc_response  # Keep reference for status
                                break

                            except Exception as e:
                                logger.warning("Failed to read response content", error=str(e))
                                continue
                        else:
                            logger.warning(f"URL approach {i+1} failed",
                                         status=doc_response.status, url=doc_url)

                except Exception as e:
                    logger.warning(f"URL approach {i+1} error", url=doc_url, error=str(e))
                    continue

            # Check if we successfully downloaded content
            if response and response.status == 200 and 'content' in locals():

                # Determine file extension based on content type and URL
                if '.txt' in successful_url:
                    file_suffix = ".txt"
                elif '/ix?' in successful_url or primary_document.endswith(('.htm', '.html')):
                    file_suffix = ".html"
                elif primary_document.endswith('.xml'):
                    file_suffix = ".xml"
                else:
                    file_suffix = ".html"  # Default to HTML for SEC documents

                # Create temporary file with correct extension
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
                    temp_file.write(content)
                    temp_file_path = temp_file.name

                document_info = {
                    "filing_type": filing["form"],
                    "title": f"{ticker_symbol} {filing['form']} Filing",
                    "date": filing["filing_date"],
                    "url": successful_url or "unknown",
                    "file_path": temp_file_path,
                    "file_size": len(content)
                }

                # Auto-process for RAG if requested
                if auto_process:
                    try:
                        # Generate unique document ID
                        document_id = str(uuid.uuid4())
                        document_name = f"{ticker_symbol}_{filing['form']}_{filing['filing_date']}"

                        # Process document
                        chunks = await document_processor.process_document(
                            file_path=temp_file_path,
                            document_name=document_name,
                            document_id=document_id
                        )

                        # Add to RAG system
                        await rag_system.add_document_chunks(
                            chunks=chunks,
                            document_id=document_id,
                            document_name=document_name,
                            file_size=len(content)
                        )

                        document_info.update({
                            "document_id": document_id,
                            "processed": True,
                            "chunks_created": len(chunks)
                        })

                        logger.info("SEC document processed for RAG",
                                  document_id=document_id,
                                  chunks=len(chunks))

                    except Exception as e:
                        logger.error("Failed to process SEC document for RAG", error=str(e))
                        document_info.update({
                            "processed": False,
                            "error": str(e)
                        })
                    finally:
                        # Clean up temporary file
                        try:
                            os.unlink(temp_file_path)
                        except:
                            pass

                return document_info

            else:
                logger.warning("Failed to download SEC document",
                             primary_document=primary_document,
                             filing_accession=filing['accession_number'])
                return None

        except Exception as e:
            logger.error("Error downloading SEC document", error=str(e))
            return None

    async def _fetch_uk_documents(
        self,
        ticker_symbol: str,
        quarter: Optional[str],
        year: int,
        filing_types: List[str],
        include_exhibits: bool,
        auto_process: bool
    ) -> Dict[str, Any]:
        """Fetch UK regulatory documents from Companies House or LSE"""

        # This is a placeholder for UK document fetching
        # You would integrate with Companies House API or other UK regulatory sources
        logger.info("UK document fetching requested", ticker=ticker_symbol)

        return {
            "success": False,
            "error": "UK document fetching not yet implemented. Please use manual upload for UK companies.",
            "documents": []
        }

    async def _fetch_international_documents(
        self,
        ticker_symbol: str,
        exchange: str,
        quarter: Optional[str],
        year: int,
        filing_types: List[str],
        include_exhibits: bool,
        auto_process: bool
    ) -> Dict[str, Any]:
        """Fetch international regulatory documents"""

        # This is a placeholder for international document fetching
        # You would integrate with relevant regulatory APIs (ESMA, SEDAR, etc.)
        logger.info("International document fetching requested",
                   ticker=ticker_symbol,
                   exchange=exchange)

        return {
            "success": False,
            "error": f"{exchange} document fetching not yet implemented. Please use manual upload for {exchange} companies.",
            "documents": []
        }


# Global instance
public_document_fetcher = PublicDocumentFetcher()

if __name__ == "__main__":
    async def test_fetch():
        async with public_document_fetcher:
            result = await public_document_fetcher.fetch_public_company_documents(
                ticker_symbol="AAPL",
                exchange="US",
                quarter='Q3',
                year=2024,
                filing_types=["10-K", "10-Q"]
            )
            print(result)

    asyncio.run(test_fetch())