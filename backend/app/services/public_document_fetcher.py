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
                        filing, ticker_symbol, auto_process
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

            async with self.session.get(url, headers=self.headers) as response:
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
            # SEC EDGAR API endpoint
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"

            async with self.session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    filings = data.get("filings", {}).get("recent", {})

                    # Filter filings by type and date
                    filtered_filings = []
                    forms = filings.get("form", [])
                    filing_dates = filings.get("filingDate", [])
                    accession_numbers = filings.get("accessionNumber", [])
                    primary_documents = filings.get("primaryDocument", [])

                    for i, form in enumerate(forms):
                        if form in filing_types:
                            filing_date = filing_dates[i] if i < len(filing_dates) else None
                            if filing_date and str(year) in filing_date:
                                # Additional quarter filtering for 10-Q
                                if quarter and form == "10-Q":
                                    # This is a simplified quarter matching - you might want to enhance this
                                    filing_month = int(filing_date.split("-")[1])
                                    quarter_months = {
                                        "Q1": [3, 4, 5],
                                        "Q2": [6, 7, 8],
                                        "Q3": [9, 10, 11],
                                        "Q4": [12, 1, 2]
                                    }
                                    if filing_month not in quarter_months.get(quarter, []):
                                        continue

                                filtered_filings.append({
                                    "form": form,
                                    "filing_date": filing_date,
                                    "accession_number": accession_numbers[i] if i < len(accession_numbers) else None,
                                    "primary_document": primary_documents[i] if i < len(primary_documents) else None
                                })

                    logger.info("Found SEC filings", count=len(filtered_filings), cik=cik)
                    return filtered_filings[:10]  # Return top 10 most recent

                else:
                    logger.error("Failed to search SEC filings", status=response.status)
                    return []

        except Exception as e:
            logger.error("Error searching SEC filings", error=str(e))
            return []

    async def _download_and_process_sec_document(
        self,
        filing: Dict[str, Any],
        ticker_symbol: str,
        auto_process: bool
    ) -> Optional[Dict[str, Any]]:
        """Download and optionally process SEC document"""
        try:
            accession_number = filing["accession_number"].replace("-", "")
            primary_document = filing["primary_document"]

            # SEC document URL
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{accession_number[:10]}/{filing['accession_number']}/{primary_document}"

            # Download document
            async with self.session.get(doc_url, headers=self.headers) as response:
                if response.status == 200:
                    content = await response.read()

                    # Create temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
                        temp_file.write(content)
                        temp_file_path = temp_file.name

                    document_info = {
                        "filing_type": filing["form"],
                        "title": f"{ticker_symbol} {filing['form']} Filing",
                        "date": filing["filing_date"],
                        "url": doc_url,
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
                                 url=doc_url,
                                 status=response.status)
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