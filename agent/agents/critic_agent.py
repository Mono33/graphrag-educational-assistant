"""
Critic Agent

Reviews lesson plans for quality and provides feedback.
"""

import json
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

from openai import AsyncOpenAI

from agent.prompts.critic_prompt import (
    get_critic_prompts,
    is_lesson_intent,
    CRITIC_SYSTEM_PROMPT,
    CRITIC_USER_TEMPLATE,
)
from agent.agents.retriever_agent import RetrievalResult

# Optional domain extensions - fails gracefully if not available
try:
    from agent.configs.domain_prompts import get_domain_extension
    DOMAIN_EXTENSIONS_AVAILABLE = True
except ImportError:
    DOMAIN_EXTENSIONS_AVAILABLE = False
    get_domain_extension = lambda d, a: ""  # Fallback: no extension

logger = logging.getLogger(__name__)


@dataclass
class CritiqueResult:
    """Result of a lesson plan critique"""
    scores: Dict[str, int]
    average_score: float
    decision: str  # "APPROVE" or "REVISE"
    strengths: list
    weaknesses: list
    revision_instructions: Optional[str]
    summary: str
    
    @property
    def approved(self) -> bool:
        return self.decision == "APPROVE"


class CriticAgent:
    """
    Critic Agent - Reviews and evaluates lesson plans.
    
    Responsibilities:
    1. Evaluate lesson plans on multiple criteria
    2. Verify content is grounded in retrieved evidence
    3. Decide whether to approve or request revision
    4. Provide specific feedback for improvements
    """
    
    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the Critic Agent.
        
        Args:
            model: OpenAI model to use for evaluation
        """
        self.model = model
        self._client: Optional[AsyncOpenAI] = None
    
    def _get_client(self) -> AsyncOpenAI:
        """Lazy initialization of OpenAI client"""
        if self._client is None:
            self._client = AsyncOpenAI()
        return self._client
    
    async def critique(
        self,
        lesson_plan: str,
        teacher_query: str,
        retrieval_result: RetrievalResult,
        revision_count: int = 0,
        max_revisions: int = 2,
        domain: str = "neuro",
        language: str = "it",
        query_intent: str = "lesson_creation"
    ) -> CritiqueResult:
        """
        Critique content and decide whether to approve.
        
        Args:
            lesson_plan: The content to review (lesson plan or informational content)
            teacher_query: Original teacher request
            retrieval_result: Retrieved context for evidence checking
            revision_count: Current revision iteration
            max_revisions: Maximum allowed revisions
            domain: Knowledge domain
            language: Language of the content
            query_intent: Detected query intent for appropriate evaluation
            
        Returns:
            CritiqueResult with scores, decision, and feedback
        """
        content_type = "lesson plan" if is_lesson_intent(query_intent) else f"{query_intent} content"
        
        logger.info(
            f"[CriticAgent] Reviewing {content_type} "
            f"(revision {revision_count}/{max_revisions})..."
        )
        
        # Auto-approve if max revisions reached
        if revision_count >= max_revisions:
            logger.info("[CriticAgent] Max revisions reached, auto-approving")
            return CritiqueResult(
                scores={"auto": 4},
                average_score=4.0,
                decision="APPROVE",
                strengths=["Multiple revision cycles completed"],
                weaknesses=[],
                revision_instructions=None,
                summary="Auto-approved after maximum revision cycles"
            )
        
        client = self._get_client()
        
        # Get intent-specific prompts
        system_prompt, user_template = get_critic_prompts(query_intent)
        
        # NEW Phase B: Add domain-specific extensions
        if DOMAIN_EXTENSIONS_AVAILABLE:
            domain_ext = get_domain_extension(domain, "critic")
            if domain_ext:
                system_prompt += domain_ext
                logger.info(f"[CriticAgent] Applied domain extension for '{domain}'")
        
        # Format retrieved context
        context_text = retrieval_result.to_context_string()
        
        # Format user prompt (handle both template types)
        if is_lesson_intent(query_intent):
            user_prompt = user_template.format(
                teacher_query=teacher_query,
                lesson_plan=lesson_plan,
                retrieved_context=context_text,
                revision_count=revision_count,
                max_revisions=max_revisions,
                domain=domain,
                language="Italian" if language == "it" else "English"
            )
        else:
            user_prompt = user_template.format(
                teacher_query=teacher_query,
                lesson_plan=lesson_plan,
                retrieved_context=context_text,
                query_intent=query_intent,
                revision_count=revision_count,
                max_revisions=max_revisions,
                domain=domain,
                language="Italian" if language == "it" else "English"
            )
        
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3  # Consistent evaluation
            )
            
            content = response.choices[0].message.content
            critique_data = json.loads(content)
            
            result = CritiqueResult(
                scores=critique_data.get("scores", {}),
                average_score=critique_data.get("average_score", 3.0),
                decision=critique_data.get("decision", "APPROVE"),
                strengths=critique_data.get("strengths", []),
                weaknesses=critique_data.get("weaknesses", []),
                revision_instructions=critique_data.get("revision_instructions"),
                summary=critique_data.get("summary", "")
            )
            
            logger.info(
                f"[CriticAgent] Decision: {result.decision} "
                f"(score: {result.average_score:.1f})"
            )
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"[CriticAgent] Failed to parse JSON response: {e}")
            # Fallback: approve
            return CritiqueResult(
                scores={},
                average_score=3.5,
                decision="APPROVE",
                strengths=[],
                weaknesses=["Could not parse critique response"],
                revision_instructions=None,
                summary="Approved due to parsing error"
            )
        except Exception as e:
            logger.error(f"[CriticAgent] Critique failed: {e}")
            raise
    
    def critique_sync(
        self,
        lesson_plan: str,
        teacher_query: str,
        retrieval_result: RetrievalResult,
        revision_count: int = 0,
        max_revisions: int = 2,
        domain: str = "neuro",
        language: str = "it"
    ) -> CritiqueResult:
        """Synchronous version of critique()"""
        import asyncio
        return asyncio.run(self.critique(
            lesson_plan, teacher_query, retrieval_result,
            revision_count, max_revisions, domain, language
        ))

