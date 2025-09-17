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

from app.core.config import settings
from app.services.document_processor import document_processor
from app.services.rag_system import rag_system

logger = structlog.get_logger(__name__)


class PublicDocumentFetcher:
    """Service for fetching public company documents from regulatory sources"""

    def __init__(self):
        self.session = None
        self.headers = {
            'User-Agent': 'Your Company Name (your.email@company.com)',  # Required by SEC
            'Accept': '*/*',
            'Host': 'www.sec.gov'
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
        """Fetch SEC EDGAR documents"""

        try:
            # Step 1: Get company CIK from ticker
            cik = await self._get_cik_from_ticker(ticker_symbol)
            if not cik:
                return {
                    "success": False,
                    "error": f"Could not find CIK for ticker {ticker_symbol}",
                    "documents": []
                }

            # Step 2: Search for filings
            filings = await self._search_sec_filings(cik, filing_types, year, quarter)
            if not filings:
                return {
                    "success": False,
                    "error": f"No filings found for {ticker_symbol} in {year}" + (f" {quarter}" if quarter else ""),
                    "documents": []
                }

            # Step 3: Download and process documents
            processed_documents = []
            for filing in filings[:10]:  # Limit to 10 documents
                try:
                    doc_result = await self._download_and_process_sec_document(
                        filing, ticker_symbol, auto_process, cik
                    )
                    if doc_result:
                        processed_documents.append(doc_result)
                except Exception as e:
                    logger.warning("Failed to process SEC filing", filing=filing, error=str(e))
                    continue

            return {
                "success": True,
                "error": None,
                "documents": processed_documents,
                "company_info": {
                    "ticker": ticker_symbol,
                    "cik": cik,
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
            url_attempts = [
                # Modern SEC data URL
                f"https://data.sec.gov/Archives/edgar/data/{cik_for_url}/{filing['accession_number']}/{primary_document}",
                # Traditional SEC archives URL
                f"https://www.sec.gov/Archives/edgar/data/{cik_for_url}/{filing['accession_number']}/{primary_document}",
                # SEC ix viewer for HTML documents
                f"https://www.sec.gov/ix?doc=/Archives/edgar/data/{cik_for_url}/{filing['accession_number']}/{primary_document}"
            ]

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

                # Determine file extension based on content type
                if '/ix?' in successful_url or primary_document.endswith(('.htm', '.html')):
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
                quarter=None,
                year=2024,
                filing_types=["10-K"]
            )
            print(result)

    asyncio.run(test_fetch())