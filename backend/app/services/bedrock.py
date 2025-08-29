import json
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
import boto3
from botocore.exceptions import ClientError
import structlog
from datetime import datetime, timedelta

from app.core.config import settings

logger = structlog.get_logger(__name__)


class BedrockService:
    """AWS Bedrock service for LLM interactions"""
    
    def __init__(self):
        self.client = None
        self.runtime_client = None
        self._daily_cost = 0.0
        self._last_reset = datetime.now().date()
        self._request_count = 0
        self._last_request_minute = datetime.now().minute
        
    async def initialize(self):
        """Initialize Bedrock clients"""
        try:
            session = boto3.Session(
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION
            )
            
            self.client = session.client('bedrock')
            self.runtime_client = session.client('bedrock-runtime')
            
            logger.info("Bedrock service initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize Bedrock service", error=str(e))
            raise
    
    def _check_rate_limits(self) -> bool:
        """Check if we're within rate limits"""
        current_minute = datetime.now().minute
        
        # Reset request count every minute
        if current_minute != self._last_request_minute:
            self._request_count = 0
            self._last_request_minute = current_minute
        
        # Check daily cost limit
        current_date = datetime.now().date()
        if current_date != self._last_reset:
            self._daily_cost = 0.0
            self._last_reset = current_date
        
        if self._request_count >= settings.BEDROCK_REQUESTS_PER_MINUTE:
            logger.warning("Rate limit exceeded", requests_per_minute=self._request_count)
            return False
        
        if self._daily_cost >= settings.BEDROCK_COST_LIMIT_DAILY:
            logger.warning("Daily cost limit exceeded", daily_cost=self._daily_cost)
            return False
        
        return True
    
    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for Claude Sonnet usage"""
        # Claude Sonnet pricing (approximate)
        input_cost_per_1k = 0.003  # $0.003 per 1K input tokens
        output_cost_per_1k = 0.015  # $0.015 per 1K output tokens
        
        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        
        return input_cost + output_cost
    
    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generate text using Claude Sonnet"""
        
        if not self._check_rate_limits():
            raise Exception("Rate limit or cost limit exceeded")
        
        if not self.runtime_client:
            await self.initialize()
        
        max_tokens = max_tokens or settings.BEDROCK_MAX_TOKENS
        temperature = temperature or settings.BEDROCK_TEMPERATURE
        
        try:
            # Prepare messages for Claude
            messages = [{"role": "user", "content": prompt}]
            
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages
            }
            
            if system_prompt:
                body["system"] = system_prompt
            
            response = self.runtime_client.invoke_model(
                modelId=settings.BEDROCK_TEXT_MODEL,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body)
            )
            
            response_body = json.loads(response.get('body').read())
            
            # Track usage and cost
            input_tokens = response_body.get('usage', {}).get('input_tokens', 0)
            output_tokens = response_body.get('usage', {}).get('output_tokens', 0)
            cost = self._estimate_cost(input_tokens, output_tokens)
            
            self._request_count += 1
            self._daily_cost += cost
            
            result = {
                'content': response_body['content'][0]['text'],
                'usage': {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens,
                    'estimated_cost': cost
                },
                'model': settings.BEDROCK_TEXT_MODEL,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(
                "Text generation completed",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost
            )
            
            return result
            
        except ClientError as e:
            logger.error("Bedrock client error", error=str(e))
            raise Exception(f"Bedrock error: {e}")
        except Exception as e:
            logger.error("Text generation failed", error=str(e))
            raise
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        
        if not self.runtime_client:
            await self.initialize()
        
        embeddings = []
        
        try:
            for text in texts:
                body = {
                    "inputText": text
                }
                
                response = self.runtime_client.invoke_model(
                    modelId=settings.BEDROCK_EMBEDDING_MODEL,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body)
                )
                
                response_body = json.loads(response.get('body').read())
                embedding = response_body.get('embedding')
                
                if embedding:
                    embeddings.append(embedding)
                else:
                    logger.warning("Empty embedding received for text", text_length=len(text))
                    embeddings.append([0.0] * 1536)  # Default embedding size
            
            logger.info("Embeddings generated", count=len(embeddings))
            return embeddings
            
        except ClientError as e:
            logger.error("Bedrock embedding error", error=str(e))
            raise Exception(f"Embedding generation failed: {e}")
        except Exception as e:
            logger.error("Embedding generation failed", error=str(e))
            raise
    
    async def stream_text_generation(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> AsyncGenerator[str, None]:
        """Stream text generation for long responses"""
        
        if not self._check_rate_limits():
            raise Exception("Rate limit or cost limit exceeded")
        
        if not self.runtime_client:
            await self.initialize()
        
        max_tokens = max_tokens or settings.BEDROCK_MAX_TOKENS
        temperature = temperature or settings.BEDROCK_TEMPERATURE
        
        try:
            messages = [{"role": "user", "content": prompt}]
            
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages
            }
            
            if system_prompt:
                body["system"] = system_prompt
            
            response = self.runtime_client.invoke_model_with_response_stream(
                modelId=settings.BEDROCK_TEXT_MODEL,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body)
            )
            
            stream = response.get('body')
            if stream:
                for event in stream:
                    chunk = event.get('chunk')
                    if chunk:
                        chunk_data = json.loads(chunk.get('bytes').decode())
                        
                        if 'delta' in chunk_data:
                            delta = chunk_data['delta']
                            if 'text' in delta:
                                yield delta['text']
                        
                        if 'usage' in chunk_data:
                            # Track final usage
                            usage = chunk_data['usage']
                            input_tokens = usage.get('input_tokens', 0)
                            output_tokens = usage.get('output_tokens', 0)
                            cost = self._estimate_cost(input_tokens, output_tokens)
                            
                            self._request_count += 1
                            self._daily_cost += cost
                            
                            logger.info(
                                "Streaming generation completed",
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                cost=cost
                            )
            
        except ClientError as e:
            logger.error("Bedrock streaming error", error=str(e))
            raise Exception(f"Streaming failed: {e}")
        except Exception as e:
            logger.error("Streaming generation failed", error=str(e))
            raise
    
    async def validate_content(
        self,
        generated_content: str,
        source_documents: List[str],
        user_prompt: str
    ) -> Dict[str, Any]:
        """Validate generated content against source documents"""
        
        validation_prompt = f"""
        Please validate the following generated content against the provided source documents and user requirements.
        
        User Requirements: {user_prompt}
        
        Generated Content: {generated_content}
        
        Source Documents:
        {' '.join(source_documents)}
        
        Please provide validation in the following JSON format:
        {{
            "overall_accuracy": "high|medium|low",
            "confidence_score": 0.0-1.0,
            "issues": [
                {{
                    "text_span": "specific text that has an issue",
                    "issue_type": "accuracy|completeness|consistency|compliance",
                    "severity": "high|medium|low",
                    "description": "detailed description of the issue",
                    "suggested_fix": "suggested correction or improvement"
                }}
            ],
            "strengths": ["list of content strengths"],
            "overall_assessment": "summary of validation results"
        }}
        """
        
        system_prompt = """You are a credit risk validation specialist. Your job is to carefully validate generated credit review content against source documents to ensure accuracy, completeness, and compliance with credit analysis standards."""
        
        try:
            result = await self.generate_text(
                prompt=validation_prompt,
                system_prompt=system_prompt,
                temperature=0.1  # Lower temperature for more consistent validation
            )
            
            # Parse the JSON response
            validation_result = json.loads(result['content'])
            validation_result['usage'] = result['usage']
            validation_result['timestamp'] = result['timestamp']
            
            return validation_result
            
        except json.JSONDecodeError as e:
            logger.error("Failed to parse validation response", error=str(e))
            return {
                "overall_accuracy": "unknown",
                "confidence_score": 0.0,
                "issues": [],
                "strengths": [],
                "overall_assessment": "Validation failed - unable to parse response",
                "error": str(e)
            }
        except Exception as e:
            logger.error("Content validation failed", error=str(e))
            raise
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        return {
            "daily_cost": self._daily_cost,
            "cost_limit": settings.BEDROCK_COST_LIMIT_DAILY,
            "requests_this_minute": self._request_count,
            "rate_limit": settings.BEDROCK_REQUESTS_PER_MINUTE,
            "last_reset_date": self._last_reset.isoformat()
        }


# Global Bedrock service instance
bedrock_service = BedrockService()