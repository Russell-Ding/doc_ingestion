import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog
from datetime import datetime
import json

from app.services.bedrock import bedrock_service
from app.services.rag_system import rag_system
from app.core.config import settings

logger = structlog.get_logger(__name__)


class ValidationSeverity(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ValidationType(Enum):
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    COMPLIANCE = "compliance"
    FACTUAL_ERROR = "factual_error"
    SOURCE_MISMATCH = "source_mismatch"
    MISSING_DATA = "missing_data"


@dataclass
class ValidationIssue:
    """Represents a validation issue with Word comment formatting"""
    text_span: str
    start_position: int
    end_position: int
    issue_type: ValidationType
    severity: ValidationSeverity
    description: str
    suggested_fix: str
    source_reference: Optional[str]
    confidence_score: float
    comment_id: str
    
    def to_word_comment(self) -> Dict[str, Any]:
        """Convert validation issue to Word comment format"""
        severity_colors = {
            ValidationSeverity.HIGH: "red",
            ValidationSeverity.MEDIUM: "orange", 
            ValidationSeverity.LOW: "yellow",
            ValidationSeverity.INFO: "blue"
        }
        
        comment_text = f"[{self.issue_type.value.upper()}] {self.description}"
        if self.suggested_fix:
            comment_text += f"\n\nSuggested Fix: {self.suggested_fix}"
        if self.source_reference:
            comment_text += f"\n\nSource: {self.source_reference}"
        
        return {
            "id": self.comment_id,
            "text": comment_text,
            "start": self.start_position,
            "end": self.end_position,
            "author": "AI Validator",
            "date": datetime.now().isoformat(),
            "severity": self.severity.value,
            "color": severity_colors[self.severity],
            "type": self.issue_type.value
        }


class ContentValidationSystem:
    """System for validating generated content with Word comment integration"""
    
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
    
    async def validate_segment_content(
        self,
        content: str,
        segment_prompt: str,
        source_documents: List[Dict[str, Any]],
        segment_name: str
    ) -> Dict[str, Any]:
        """Validate generated content and return structured results"""
        
        logger.info(
            "Starting content validation",
            content_length=len(content),
            segment_name=segment_name
        )
        
        try:
            # Run multiple validation checks in parallel
            validation_tasks = [
                self._validate_accuracy(content, source_documents),
                self._validate_completeness(content, segment_prompt),
                self._validate_consistency(content, source_documents),
                self._validate_factual_claims(content, source_documents),
                self._validate_compliance(content, segment_name)
            ]
            
            validation_results = await asyncio.gather(*validation_tasks)
            
            # Aggregate results
            all_issues = []
            for result in validation_results:
                if result and result.get("issues"):
                    all_issues.extend(result["issues"])
            
            # Convert to ValidationIssue objects
            structured_issues = []
            for i, issue in enumerate(all_issues):
                validation_issue = ValidationIssue(
                    text_span=issue.get("text_span", ""),
                    start_position=self._find_text_position(content, issue.get("text_span", "")),
                    end_position=self._find_text_position(content, issue.get("text_span", "")) + len(issue.get("text_span", "")),
                    issue_type=ValidationType(issue.get("issue_type", "accuracy")),
                    severity=ValidationSeverity(issue.get("severity", "medium")),
                    description=issue.get("description", ""),
                    suggested_fix=issue.get("suggested_fix", ""),
                    source_reference=issue.get("source_reference", None),
                    confidence_score=issue.get("confidence_score", 0.5),
                    comment_id=f"validation_{segment_name}_{i+1}"
                )
                structured_issues.append(validation_issue)
            
            # Calculate overall quality score
            quality_score = self._calculate_quality_score(structured_issues)
            
            # Generate summary
            summary = self._generate_validation_summary(structured_issues, quality_score)
            
            return {
                "validation_id": f"val_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "segment_name": segment_name,
                "overall_quality_score": quality_score,
                "total_issues": len(structured_issues),
                "issues_by_severity": self._count_issues_by_severity(structured_issues),
                "issues_by_type": self._count_issues_by_type(structured_issues),
                "validation_issues": [issue.__dict__ for issue in structured_issues],
                "word_comments": [issue.to_word_comment() for issue in structured_issues],
                "summary": summary,
                "validation_timestamp": datetime.now().isoformat(),
                "passed_validation": quality_score >= settings.VALIDATION_CONFIDENCE_THRESHOLD
            }
            
        except Exception as e:
            logger.error("Content validation failed", error=str(e))
            return {
                "validation_id": f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "segment_name": segment_name,
                "overall_quality_score": 0.0,
                "total_issues": 0,
                "error": str(e),
                "validation_timestamp": datetime.now().isoformat(),
                "passed_validation": False
            }
    
    async def _validate_accuracy(
        self,
        content: str,
        source_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate accuracy of content against source documents"""
        
        # Extract numerical claims and facts from content
        numerical_claims = self._extract_numerical_claims(content)
        factual_claims = self._extract_factual_claims(content)
        
        issues = []
        
        # Validate numerical claims
        for claim in numerical_claims:
            validation_result = await self._validate_numerical_claim(
                claim, source_documents
            )
            if not validation_result["valid"]:
                issues.append({
                    "text_span": claim["text"],
                    "issue_type": "accuracy",
                    "severity": "high" if claim["importance"] > 0.7 else "medium",
                    "description": f"Numerical claim could not be verified: {validation_result['reason']}",
                    "suggested_fix": "Verify the number against source documents or remove if uncertain",
                    "confidence_score": validation_result["confidence"]
                })
        
        # Validate factual claims
        for claim in factual_claims:
            validation_result = await self._validate_factual_claim(
                claim, source_documents
            )
            if not validation_result["valid"]:
                issues.append({
                    "text_span": claim["text"],
                    "issue_type": "factual_error",
                    "severity": "high",
                    "description": f"Factual claim not supported by sources: {validation_result['reason']}",
                    "suggested_fix": "Revise based on source documentation or add qualification",
                    "source_reference": validation_result.get("conflicting_source"),
                    "confidence_score": validation_result["confidence"]
                })
        
        return {"issues": issues}
    
    async def _validate_completeness(
        self,
        content: str,
        segment_prompt: str
    ) -> Dict[str, Any]:
        """Check if content addresses all requirements in the prompt"""
        
        validation_prompt = f"""
        Evaluate if the following generated content fully addresses the user's requirements.
        
        User Requirements: {segment_prompt}
        
        Generated Content: {content}
        
        Please identify any missing elements or requirements that were not addressed.
        Respond in JSON format:
        {{
            "completeness_score": 0.0-1.0,
            "missing_elements": [
                {{
                    "requirement": "specific requirement not addressed",
                    "severity": "high|medium|low",
                    "description": "explanation of what's missing",
                    "suggested_addition": "what should be added"
                }}
            ],
            "well_covered_elements": ["list of well-addressed requirements"]
        }}
        """
        
        try:
            result = await bedrock_service.generate_text(
                prompt=validation_prompt,
                temperature=0.1
            )
            
            analysis = json.loads(result['content'])
            
            issues = []
            for missing in analysis.get("missing_elements", []):
                issues.append({
                    "text_span": "",  # No specific text span for missing content
                    "issue_type": "completeness",
                    "severity": missing.get("severity", "medium"),
                    "description": f"Missing requirement: {missing.get('description', '')}",
                    "suggested_fix": missing.get("suggested_addition", ""),
                    "confidence_score": analysis.get("completeness_score", 0.5)
                })
            
            return {"issues": issues}
            
        except Exception as e:
            logger.warning("Completeness validation failed", error=str(e))
            return {"issues": []}
    
    async def _validate_consistency(
        self,
        content: str,
        source_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check for internal consistency and consistency with sources"""
        
        issues = []
        
        # Check for contradictory statements within the content
        sentences = self._split_into_sentences(content)
        
        for i, sentence1 in enumerate(sentences):
            for j, sentence2 in enumerate(sentences[i+1:], i+1):
                if self._are_potentially_contradictory(sentence1, sentence2):
                    issues.append({
                        "text_span": sentence1,
                        "issue_type": "consistency",
                        "severity": "medium",
                        "description": f"Potential contradiction with: '{sentence2}'",
                        "suggested_fix": "Review and resolve the contradiction",
                        "confidence_score": 0.6
                    })
        
        return {"issues": issues}
    
    async def _validate_factual_claims(
        self,
        content: str,
        source_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate specific factual claims against source documents"""
        
        validation_prompt = f"""
        Please validate the factual claims in the following content against the provided source documents.
        
        Content to validate: {content}
        
        Source documents: {json.dumps([doc.get("chunks", [])[:3] for doc in source_documents[:5]])}
        
        For each factual claim, check if it's supported by the sources.
        Respond in JSON format:
        {{
            "validated_claims": [
                {{
                    "claim_text": "specific claim from content",
                    "supported": true/false,
                    "source_reference": "reference to supporting source",
                    "confidence": 0.0-1.0,
                    "issue_description": "if not supported, explain why"
                }}
            ]
        }}
        """
        
        try:
            result = await bedrock_service.generate_text(
                prompt=validation_prompt,
                temperature=0.1
            )
            
            analysis = json.loads(result['content'])
            
            issues = []
            for claim in analysis.get("validated_claims", []):
                if not claim.get("supported", True):
                    issues.append({
                        "text_span": claim.get("claim_text", ""),
                        "issue_type": "source_mismatch",
                        "severity": "high",
                        "description": claim.get("issue_description", "Claim not supported by sources"),
                        "suggested_fix": "Verify claim against source documents or remove",
                        "source_reference": claim.get("source_reference"),
                        "confidence_score": claim.get("confidence", 0.5)
                    })
            
            return {"issues": issues}
            
        except Exception as e:
            logger.warning("Factual claim validation failed", error=str(e))
            return {"issues": []}
    
    async def _validate_compliance(
        self,
        content: str,
        segment_name: str
    ) -> Dict[str, Any]:
        """Check compliance with credit analysis standards"""
        
        compliance_rules = self.validation_rules.get("compliance", {})
        issues = []
        
        # Check for required disclosures
        required_disclosures = compliance_rules.get("required_disclosures", [])
        for disclosure in required_disclosures:
            if not self._contains_disclosure(content, disclosure):
                issues.append({
                    "text_span": "",
                    "issue_type": "compliance",
                    "severity": "high",
                    "description": f"Missing required disclosure: {disclosure['name']}",
                    "suggested_fix": f"Add disclosure: {disclosure['text']}",
                    "confidence_score": 0.9
                })
        
        # Check for prohibited terms or statements
        prohibited_terms = compliance_rules.get("prohibited_terms", [])
        for term in prohibited_terms:
            if term.lower() in content.lower():
                issues.append({
                    "text_span": self._find_term_in_content(content, term),
                    "issue_type": "compliance",
                    "severity": "medium",
                    "description": f"Use of prohibited term: {term}",
                    "suggested_fix": "Replace with appropriate alternative",
                    "confidence_score": 0.8
                })
        
        return {"issues": issues}
    
    def _extract_numerical_claims(self, content: str) -> List[Dict[str, Any]]:
        """Extract numerical claims from content"""
        patterns = [
            r'\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|thousand))?',
            r'\d+(?:\.\d+)?%',
            r'\d+(?:\.\d+)?(?:\s*(?:million|billion|thousand|times|x))',
            r'ratio\s+of\s+[\d.]+',
            r'increased?\s+by\s+[\d.]+%?',
            r'decreased?\s+by\s+[\d.]+%?'
        ]
        
        claims = []
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                claims.append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "importance": 0.8  # Default importance
                })
        
        return claims
    
    def _extract_factual_claims(self, content: str) -> List[Dict[str, Any]]:
        """Extract factual claims from content"""
        # Simple heuristic to identify potential factual claims
        sentences = self._split_into_sentences(content)
        factual_indicators = [
            'according to', 'based on', 'the company', 'reported', 'stated',
            'indicated', 'shows', 'demonstrates', 'evidenced by'
        ]
        
        claims = []
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in factual_indicators):
                claims.append({
                    "text": sentence.strip(),
                    "importance": 0.7
                })
        
        return claims
    
    async def _validate_numerical_claim(
        self,
        claim: Dict[str, Any],
        source_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate a specific numerical claim"""
        # Simplified validation - in practice, this would involve more sophisticated
        # numerical extraction and comparison
        
        claim_text = claim["text"]
        
        # Search for similar numbers in source documents
        for doc in source_documents:
            for chunk in doc.get("chunks", []):
                if claim_text.lower() in chunk.get("content", "").lower():
                    return {"valid": True, "confidence": 0.9}
        
        return {
            "valid": False, 
            "reason": "Number not found in source documents",
            "confidence": 0.8
        }
    
    async def _validate_factual_claim(
        self,
        claim: Dict[str, Any],
        source_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate a specific factual claim"""
        # Simplified validation logic
        claim_text = claim["text"].lower()
        
        # Check if similar content exists in source documents
        for doc in source_documents:
            for chunk in doc.get("chunks", []):
                chunk_content = chunk.get("content", "").lower()
                # Simple similarity check - in practice, use more sophisticated methods
                words_in_common = len(set(claim_text.split()) & set(chunk_content.split()))
                if words_in_common > 3:
                    return {"valid": True, "confidence": 0.7}
        
        return {
            "valid": False,
            "reason": "Claim not supported by available source documents",
            "confidence": 0.6
        }
    
    def _find_text_position(self, content: str, text_span: str) -> int:
        """Find the position of text span in content"""
        if not text_span:
            return 0
        position = content.find(text_span)
        return position if position != -1 else 0
    
    def _split_into_sentences(self, content: str) -> List[str]:
        """Split content into sentences"""
        # Simple sentence splitting - in practice, use more sophisticated NLP
        sentences = re.split(r'[.!?]+', content)
        return [s.strip() for s in sentences if s.strip()]
    
    def _are_potentially_contradictory(self, sentence1: str, sentence2: str) -> bool:
        """Check if two sentences might be contradictory"""
        # Simplified contradiction detection
        contradiction_pairs = [
            ('increase', 'decrease'), ('improve', 'deteriorate'),
            ('strong', 'weak'), ('high', 'low'), ('positive', 'negative')
        ]
        
        s1_lower = sentence1.lower()
        s2_lower = sentence2.lower()
        
        for word1, word2 in contradiction_pairs:
            if word1 in s1_lower and word2 in s2_lower:
                return True
        
        return False
    
    def _calculate_quality_score(self, issues: List[ValidationIssue]) -> float:
        """Calculate overall quality score based on issues"""
        if not issues:
            return 1.0
        
        severity_weights = {
            ValidationSeverity.HIGH: 0.3,
            ValidationSeverity.MEDIUM: 0.15,
            ValidationSeverity.LOW: 0.05,
            ValidationSeverity.INFO: 0.01
        }
        
        total_deduction = sum(
            severity_weights.get(issue.severity, 0.1) for issue in issues
        )
        
        return max(0.0, 1.0 - total_deduction)
    
    def _count_issues_by_severity(self, issues: List[ValidationIssue]) -> Dict[str, int]:
        """Count issues by severity level"""
        counts = {severity.value: 0 for severity in ValidationSeverity}
        for issue in issues:
            counts[issue.severity.value] += 1
        return counts
    
    def _count_issues_by_type(self, issues: List[ValidationIssue]) -> Dict[str, int]:
        """Count issues by validation type"""
        counts = {vtype.value: 0 for vtype in ValidationType}
        for issue in issues:
            counts[issue.issue_type.value] += 1
        return counts
    
    def _generate_validation_summary(
        self,
        issues: List[ValidationIssue],
        quality_score: float
    ) -> Dict[str, Any]:
        """Generate a summary of validation results"""
        
        if not issues:
            return {
                "overall_assessment": "Content passed validation with no issues found.",
                "quality_rating": "Excellent",
                "key_strengths": ["Accurate information", "Complete coverage", "Consistent messaging"],
                "priority_actions": []
            }
        
        high_priority_issues = [i for i in issues if i.severity == ValidationSeverity.HIGH]
        medium_priority_issues = [i for i in issues if i.severity == ValidationSeverity.MEDIUM]
        
        quality_ratings = {
            (0.9, 1.0): "Excellent",
            (0.8, 0.9): "Good", 
            (0.6, 0.8): "Fair",
            (0.0, 0.6): "Poor"
        }
        
        quality_rating = "Poor"
        for (min_score, max_score), rating in quality_ratings.items():
            if min_score <= quality_score < max_score:
                quality_rating = rating
                break
        
        return {
            "overall_assessment": f"Content validation completed with {len(issues)} issues found. Quality score: {quality_score:.2f}",
            "quality_rating": quality_rating,
            "priority_actions": [
                f"Address {len(high_priority_issues)} high-priority issues" if high_priority_issues else None,
                f"Review {len(medium_priority_issues)} medium-priority issues" if medium_priority_issues else None
            ],
            "most_common_issue_type": max(
                self._count_issues_by_type(issues).items(),
                key=lambda x: x[1]
            )[0] if issues else None
        }
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules and compliance requirements"""
        return {
            "compliance": {
                "required_disclosures": [
                    {
                        "name": "Risk Disclaimer",
                        "text": "This analysis is based on available information and involves inherent uncertainties.",
                        "required_for": ["risk_assessment", "financial_analysis"]
                    }
                ],
                "prohibited_terms": [
                    "guaranteed", "risk-free", "certain", "impossible to lose"
                ]
            }
        }
    
    def _contains_disclosure(self, content: str, disclosure: Dict[str, Any]) -> bool:
        """Check if content contains required disclosure"""
        # Simplified check - look for key phrases
        key_phrases = disclosure.get("key_phrases", [disclosure["name"].lower()])
        content_lower = content.lower()
        
        return any(phrase.lower() in content_lower for phrase in key_phrases)
    
    def _find_term_in_content(self, content: str, term: str) -> str:
        """Find and return the sentence containing the term"""
        sentences = self._split_into_sentences(content)
        for sentence in sentences:
            if term.lower() in sentence.lower():
                return sentence
        return term


# Global validation system instance
validation_system = ContentValidationSystem()