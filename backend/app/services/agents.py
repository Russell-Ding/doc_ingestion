import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog
from datetime import datetime

from app.services.bedrock import bedrock_service
from app.services.rag_system import rag_system, RetrievalResult
from app.core.config import settings

logger = structlog.get_logger(__name__)


class AgentType(Enum):
    """Types of agents in the system"""
    DOCUMENT_FINDER = "document_finder"
    CONTENT_GENERATOR = "content_generator"
    VALIDATOR = "validator"
    REPORT_COORDINATOR = "report_coordinator"


@dataclass
class AgentResult:
    """Result returned by an agent"""
    agent_type: AgentType
    success: bool
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None


class BaseAgent:
    """Base class for all agents"""
    
    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type
        self.logger = structlog.get_logger(f"agent.{agent_type.value}")
    
    async def execute(self, task_data: Dict[str, Any]) -> AgentResult:
        """Execute the agent's task"""
        start_time = datetime.now()
        
        try:
            result_data = await self._execute_task(task_data)
            
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return AgentResult(
                agent_type=self.agent_type,
                success=True,
                data=result_data,
                metadata={"execution_time_ms": execution_time},
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            self.logger.error(
                "Agent execution failed",
                error=str(e),
                task_data=task_data,
                execution_time_ms=execution_time
            )
            
            return AgentResult(
                agent_type=self.agent_type,
                success=False,
                data={},
                metadata={"execution_time_ms": execution_time},
                error_message=str(e),
                execution_time_ms=execution_time
            )
    
    async def _execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Override this method in subclasses"""
        raise NotImplementedError("Subclasses must implement _execute_task")


class DocumentFinderAgent(BaseAgent):
    """Agent responsible for finding relevant documents for report segments"""
    
    def __init__(self):
        super().__init__(AgentType.DOCUMENT_FINDER)
    
    async def _execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Find relevant documents for a segment"""
        
        segment_prompt = task_data.get("segment_prompt", "")
        required_document_types = task_data.get("required_document_types", [])
        max_documents = task_data.get("max_documents", 10)
        selected_document_ids = task_data.get("selected_document_ids", None)
        
        if not segment_prompt:
            raise ValueError("segment_prompt is required")
        
        self.logger.info(
            "Finding relevant documents",
            segment_prompt_length=len(segment_prompt),
            required_types=required_document_types,
            selected_document_ids=selected_document_ids
        )
        
        # Analyze the prompt to understand document requirements
        document_analysis = await self._analyze_document_requirements(segment_prompt)
        
        # Retrieve relevant chunks based on the analysis
        try:
            retrieval_results = await rag_system.retrieve_relevant_chunks(
                query=segment_prompt,
                document_types=required_document_types if required_document_types else None,
                max_results=max_documents,
                include_tables=document_analysis.get("needs_tables", True),
                focus_keywords=document_analysis.get("focus_keywords", []),
                document_ids=selected_document_ids  # Pass the selected document IDs to filter
            )
        except Exception as e:
            self.logger.warning(f"Document retrieval failed: {e}, proceeding with empty results")
            retrieval_results = []
        
        # Group results by document
        documents_found = {}
        for result in retrieval_results:
            doc_id = result.document_id
            if doc_id not in documents_found:
                documents_found[doc_id] = {
                    "document_id": doc_id,
                    "chunks": [],
                    "total_relevance_score": 0.0,
                    "chunk_types": set()
                }
            
            documents_found[doc_id]["chunks"].append({
                "chunk_id": result.chunk_id,
                "content": result.content,
                "chunk_type": result.chunk_type,
                "page_number": result.page_number,
                "section_title": result.section_title,
                "relevance_score": result.relevance_score,
                "retrieval_method": result.retrieval_method
            })
            
            documents_found[doc_id]["total_relevance_score"] += result.relevance_score
            documents_found[doc_id]["chunk_types"].add(result.chunk_type)
        
        # Convert chunk_types set to list for JSON serialization
        for doc_data in documents_found.values():
            doc_data["chunk_types"] = list(doc_data["chunk_types"])
        
        # Rank documents by total relevance
        ranked_documents = sorted(
            documents_found.values(),
            key=lambda x: x["total_relevance_score"],
            reverse=True
        )
        
        if not ranked_documents:
            self.logger.warning("No relevant documents found for segment", prompt=segment_prompt[:100])
        
        return {
            "documents_found": ranked_documents,
            "total_documents": len(ranked_documents),
            "total_chunks": len(retrieval_results),
            "document_analysis": document_analysis,
            "retrieval_summary": {
                "semantic_matches": len([r for r in retrieval_results if r.retrieval_method == "semantic"]),
                "keyword_matches": len([r for r in retrieval_results if r.retrieval_method == "keyword"]),
                "table_matches": len([r for r in retrieval_results if r.retrieval_method == "table_specific"])
            }
        }
    
    async def _analyze_document_requirements(self, segment_prompt: str) -> Dict[str, Any]:
        """Analyze the segment prompt to understand document requirements"""
        
        analysis_prompt = f"""
        Analyze the following credit review report segment prompt and determine what types of documents and information would be most relevant.
        
        Segment Prompt: {segment_prompt}
        
        Please provide analysis in JSON format:
        {{
            "needs_tables": true/false,
            "focus_keywords": ["keyword1", "keyword2", ...],
            "document_types_needed": ["financial_statements", "contracts", "reports", etc.],
            "content_priority": "numerical|textual|mixed",
            "specific_requirements": ["requirement1", "requirement2", ...],
            "expected_sections": ["section1", "section2", ...]
        }}
        
        Focus on credit risk analysis, financial data, and business documentation.
        """
        
        system_prompt = """You are a credit risk analysis specialist. Your job is to understand what types of documents and data would be needed to fulfill a report segment request."""
        
        try:
            result = await bedrock_service.generate_text(
                prompt=analysis_prompt,
                system_prompt=system_prompt,
                temperature=0.1
            )
            
            # Extract content from the response
            raw_content = result.get('content', '') if result else ''
            
            if not raw_content:
                self.logger.warning("Empty response from LLM for document analysis")
                raise ValueError("Empty response")
            
            # Try to extract JSON from the response (in case it has extra text)
            json_start = raw_content.find('{')
            json_end = raw_content.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                self.logger.warning("No JSON found in document analysis response", response=raw_content[:200])
                raise ValueError("No valid JSON found")
            
            json_content = raw_content[json_start:json_end]
            
            # Parse the JSON response
            analysis = json.loads(json_content)
            
            # Validate that required keys exist
            required_keys = ["needs_tables", "focus_keywords", "document_types_needed", "content_priority"]
            for key in required_keys:
                if key not in analysis:
                    analysis[key] = self._get_default_value(key, segment_prompt)
            
            return analysis
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning("Failed to parse document analysis", error=str(e), response=raw_content[:200] if 'raw_content' in locals() else "No content")
            # Fallback analysis
            return {
                "needs_tables": "financial" in segment_prompt.lower() or "data" in segment_prompt.lower(),
                "focus_keywords": [],
                "document_types_needed": [],
                "content_priority": "mixed",
                "specific_requirements": [],
                "expected_sections": []
            }
        except Exception as e:
            self.logger.error("Document requirement analysis failed", error=str(e))
            return {
                "needs_tables": True,
                "focus_keywords": [],
                "document_types_needed": [],
                "content_priority": "mixed",
                "specific_requirements": [],
                "expected_sections": []
            }
    
    def _get_default_value(self, key: str, segment_prompt: str):
        """Get default value for missing analysis keys"""
        defaults = {
            "needs_tables": "financial" in segment_prompt.lower() or "data" in segment_prompt.lower(),
            "focus_keywords": [],
            "document_types_needed": [],
            "content_priority": "mixed",
            "specific_requirements": [],
            "expected_sections": []
        }
        return defaults.get(key, None)


class ContentGeneratorAgent(BaseAgent):
    """Agent responsible for generating report content"""
    
    def __init__(self):
        super().__init__(AgentType.CONTENT_GENERATOR)
    
    async def _execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content for a report segment"""
        
        segment_prompt = task_data.get("segment_prompt", "")
        segment_name = task_data.get("segment_name", "")
        retrieved_documents = task_data.get("retrieved_documents", [])
        generation_settings = task_data.get("generation_settings", {})
        
        if not segment_prompt:
            raise ValueError("segment_prompt is required")
        
        if not retrieved_documents:
            self.logger.warning("No documents retrieved for content generation, proceeding with general knowledge")
            # Instead of failing, generate content with limited context
            retrieved_documents = []
        
        self.logger.info(
            "Generating content",
            segment_name=segment_name,
            document_count=len(retrieved_documents),
            prompt_length=len(segment_prompt)
        )
        
        # Prepare context from retrieved documents
        if retrieved_documents:
            context = await self._prepare_document_context(retrieved_documents)
        else:
            context = "No specific documents were retrieved for this segment. Please generate content based on general knowledge and best practices."
        
        # Generate content using the context
        generated_content = await self._generate_content_with_context(
            segment_prompt,
            segment_name,
            context,
            generation_settings
        )
        
        # Extract references used in generation
        references = self._extract_references(retrieved_documents, generated_content)
        
        return {
            "generated_content": generated_content["content"],
            "usage_stats": generated_content["usage"],
            "references": references,
            "context_summary": {
                "total_documents": len(retrieved_documents),
                "total_chunks": sum(len(doc.get("chunks", [])) for doc in retrieved_documents),
                "context_length": len(context),
                "primary_sources": [doc["document_id"] for doc in retrieved_documents[:3]] if retrieved_documents else []
            }
        }
    
    async def _prepare_document_context(self, retrieved_documents: List[Dict[str, Any]]) -> str:
        """Prepare context string from retrieved documents"""
        
        context_parts = []
        
        for doc in retrieved_documents:
            context_parts.append(f"=== Document: {doc['document_id']} ===\n")
            
            for chunk in doc["chunks"]:
                chunk_header = f"[{chunk['chunk_type']}"
                if chunk.get("page_number"):
                    chunk_header += f", Page {chunk['page_number']}"
                if chunk.get("section_title"):
                    chunk_header += f", Section: {chunk['section_title']}"
                chunk_header += f", Relevance: {chunk['relevance_score']:.2f}]"
                
                context_parts.append(f"{chunk_header}\n{chunk['content']}\n")
            
            context_parts.append("\n" + "="*50 + "\n")
        
        return "\n".join(context_parts)
    
    async def _generate_content_with_context(
        self,
        segment_prompt: str,
        segment_name: str,
        context: str,
        generation_settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate content using document context"""
        
        generation_prompt = f"""
        You are writing a section for a credit review report. Please generate content based on the provided document context and user requirements.
        
        Section Name: {segment_name}
        User Requirements: {segment_prompt}
        
        Document Context:
        {context}
        
        Please generate a comprehensive section that:
        1. Directly addresses the user's requirements
        2. Uses specific information from the provided documents
        3. Maintains professional credit analysis tone
        4. Includes relevant numerical data and facts
        5. Properly references the source materials
        6. Follows standard credit review report format
        
        Generate ONLY the section content - do not include explanations or meta-commentary.
        """
        
        system_prompt = """You are an expert credit risk analyst with extensive experience in writing credit review reports. Your analysis should be thorough, accurate, and well-supported by the provided documentation."""
        
        max_tokens = generation_settings.get("max_tokens", settings.BEDROCK_MAX_TOKENS)
        temperature = generation_settings.get("temperature", settings.BEDROCK_TEMPERATURE)
        
        result = await bedrock_service.generate_text(
            prompt=generation_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return result
    
    def _extract_references(
        self,
        retrieved_documents: List[Dict[str, Any]],
        generated_content: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract and format references used in content generation"""
        
        references = []
        content_text = generated_content["content"].lower()
        
        for doc in retrieved_documents:
            doc_referenced = False
            referenced_chunks = []
            
            for chunk in doc["chunks"]:
                # Simple heuristic to check if chunk content was likely used
                chunk_words = chunk["content"].lower().split()[:20]  # First 20 words
                
                # Check if significant portion of chunk appears in generated content
                word_matches = sum(1 for word in chunk_words if word in content_text)
                if word_matches >= 3:  # At least 3 words match
                    doc_referenced = True
                    referenced_chunks.append({
                        "chunk_id": chunk["chunk_id"],
                        "chunk_type": chunk["chunk_type"],
                        "page_number": chunk.get("page_number"),
                        "section_title": chunk.get("section_title"),
                        "relevance_score": chunk["relevance_score"]
                    })
            
            if doc_referenced:
                references.append({
                    "document_id": doc["document_id"],
                    "chunks_used": referenced_chunks,
                    "reference_strength": len(referenced_chunks) / len(doc["chunks"])
                })
        
        return references


class ValidatorAgent(BaseAgent):
    """Agent responsible for validating generated content"""
    
    def __init__(self):
        super().__init__(AgentType.VALIDATOR)
    
    async def _execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate generated content against source documents"""
        
        generated_content = task_data.get("generated_content", "")
        segment_prompt = task_data.get("segment_prompt", "")
        source_documents = task_data.get("source_documents", [])
        
        if not generated_content:
            raise ValueError("generated_content is required")
        
        if not source_documents:
            self.logger.warning("No source documents provided for validation, skipping document-based validation")
            # Return a basic validation result when no source documents are available
            return {
                "overall_accuracy": "unknown",
                "confidence_score": 0.5,
                "issues": [],
                "strengths": ["Content generated without specific source validation"],
                "overall_assessment": "Generated content without source document validation",
                "validation_metadata": {
                    "total_issues": 0,
                    "high_severity_issues": 0,
                    "medium_severity_issues": 0,
                    "low_severity_issues": 0,
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        self.logger.info(
            "Validating generated content",
            content_length=len(generated_content),
            source_document_count=len(source_documents)
        )
        
        # Prepare source context
        source_context = await self._prepare_source_context(source_documents)
        
        # Perform validation
        validation_result = await bedrock_service.validate_content(
            generated_content=generated_content,
            source_documents=[source_context],
            user_prompt=segment_prompt
        )
        
        # Process validation result into structured format
        processed_validation = await self._process_validation_result(
            validation_result,
            generated_content
        )
        
        return processed_validation
    
    async def _prepare_source_context(self, source_documents: List[Dict[str, Any]]) -> str:
        """Prepare source context for validation"""
        
        context_parts = []
        
        for doc in source_documents:
            for chunk in doc.get("chunks", []):
                context_parts.append(f"Source: {chunk['chunk_type']}")
                if chunk.get("page_number"):
                    context_parts.append(f"Page: {chunk['page_number']}")
                context_parts.append(f"Content: {chunk['content']}")
                context_parts.append("---")
        
        return "\n".join(context_parts)
    
    async def _process_validation_result(
        self,
        validation_result: Dict[str, Any],
        generated_content: str
    ) -> Dict[str, Any]:
        """Process and structure validation results"""
        
        # Extract validation issues and create structured comments
        issues = validation_result.get("issues", [])
        processed_issues = []
        
        for issue in issues:
            text_span = issue.get("text_span", "")
            
            # Find the position of the text span in the content
            start_pos = generated_content.lower().find(text_span.lower())
            end_pos = start_pos + len(text_span) if start_pos != -1 else -1
            
            processed_issue = {
                "text_span": text_span,
                "start_position": start_pos,
                "end_position": end_pos,
                "issue_type": issue.get("issue_type", "unknown"),
                "severity": issue.get("severity", "medium"),
                "description": issue.get("description", ""),
                "suggested_fix": issue.get("suggested_fix", ""),
                "validation_id": f"val_{len(processed_issues) + 1}"
            }
            
            processed_issues.append(processed_issue)
        
        return {
            "overall_accuracy": validation_result.get("overall_accuracy", "unknown"),
            "confidence_score": validation_result.get("confidence_score", 0.0),
            "issues": processed_issues,
            "strengths": validation_result.get("strengths", []),
            "overall_assessment": validation_result.get("overall_assessment", ""),
            "validation_metadata": {
                "total_issues": len(processed_issues),
                "high_severity_issues": len([i for i in processed_issues if i["severity"] == "high"]),
                "medium_severity_issues": len([i for i in processed_issues if i["severity"] == "medium"]),
                "low_severity_issues": len([i for i in processed_issues if i["severity"] == "low"]),
                "timestamp": datetime.now().isoformat()
            }
        }


class ReportCoordinatorAgent(BaseAgent):
    """Agent responsible for coordinating the entire report generation process"""
    
    def __init__(self):
        super().__init__(AgentType.REPORT_COORDINATOR)
        self.document_finder = DocumentFinderAgent()
        self.content_generator = ContentGeneratorAgent()
        self.validator = ValidatorAgent()
    
    async def _execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate the generation of a complete report segment"""
        
        segment_data = task_data.get("segment_data", {})
        validation_enabled = task_data.get("validation_enabled", True)
        selected_document_ids = task_data.get("selected_document_ids", None)
        
        if not segment_data:
            raise ValueError("segment_data is required")
        
        segment_name = segment_data.get("name", "")
        segment_prompt = segment_data.get("prompt", "")
        
        self.logger.info(
            "Coordinating segment generation",
            segment_name=segment_name,
            validation_enabled=validation_enabled,
            selected_document_ids=selected_document_ids
        )
        
        # Step 1: Find relevant documents
        self.logger.info("Step 1: Finding relevant documents")
        doc_finder_result = await self.document_finder.execute({
            "segment_prompt": segment_prompt,
            "required_document_types": segment_data.get("required_document_types", []),
            "max_documents": segment_data.get("max_documents", 10),
            "selected_document_ids": selected_document_ids
        })
        
        if not doc_finder_result.success:
            raise Exception(f"Document finding failed: {doc_finder_result.error_message}")
        
        # Step 2: Generate content
        self.logger.info("Step 2: Generating content")
        content_result = await self.content_generator.execute({
            "segment_prompt": segment_prompt,
            "segment_name": segment_name,
            "retrieved_documents": doc_finder_result.data["documents_found"],
            "generation_settings": segment_data.get("generation_settings", {})
        })
        
        if not content_result.success:
            raise Exception(f"Content generation failed: {content_result.error_message}")
        
        # Step 3: Validate content (if enabled)
        validation_result = None
        if validation_enabled and settings.ENABLE_CONTENT_VALIDATION:
            self.logger.info("Step 3: Validating content")
            validation_result = await self.validator.execute({
                "generated_content": content_result.data["generated_content"],
                "segment_prompt": segment_prompt,
                "source_documents": doc_finder_result.data["documents_found"]
            })
            
            if not validation_result.success:
                self.logger.warning(
                    "Content validation failed",
                    error=validation_result.error_message
                )
        
        # Combine results
        return {
            "segment_name": segment_name,
            "generated_content": content_result.data["generated_content"],
            "document_retrieval": doc_finder_result.data,
            "content_generation": {
                "usage_stats": content_result.data["usage_stats"],
                "references": content_result.data["references"],
                "context_summary": content_result.data["context_summary"]
            },
            "validation": validation_result.data if validation_result and validation_result.success else None,
            "execution_summary": {
                "document_finding_time_ms": doc_finder_result.execution_time_ms,
                "content_generation_time_ms": content_result.execution_time_ms,
                "validation_time_ms": validation_result.execution_time_ms if validation_result else 0,
                "total_time_ms": (
                    doc_finder_result.execution_time_ms +
                    content_result.execution_time_ms +
                    (validation_result.execution_time_ms if validation_result else 0)
                )
            }
        }


# Global agent instances
document_finder_agent = DocumentFinderAgent()
content_generator_agent = ContentGeneratorAgent()
validator_agent = ValidatorAgent()
report_coordinator_agent = ReportCoordinatorAgent()