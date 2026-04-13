"""
Planner Agent

Analyzes teacher queries and creates structured retrieval plans.
Determines what to search in the knowledge graph.
"""

import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from openai import AsyncOpenAI
from config import config as app_config, extract_response_content

from agent.prompts.planner_prompt import PLANNER_SYSTEM_PROMPT, PLANNER_USER_TEMPLATE

logger = logging.getLogger(__name__)


@dataclass
class RetrievalPlan:
    """Structured plan for what to retrieve"""
    query_intent: str  # lesson_creation, definition, comparison, etc.
    key_concepts: List[str]
    search_queries: List[str]
    lesson_type: Optional[str] = None  # Only for lesson/activity intents
    target_grade: Optional[str] = None
    special_needs: Optional[List[str]] = None
    time_constraints: Optional[str] = None
    intent_confidence: str = "MEDIUM"
    reasoning: Optional[str] = None
    
    # NEW Phase A: Scope detection fields
    scope_status: str = "in_scope"  # in_scope, partial_scope, out_of_scope
    scope_confidence: float = 1.0   # 0.0-1.0 confidence in scope detection
    subject_concepts: Optional[List[str]] = None  # Subject-specific (may need external APIs)
    pedagogy_concepts: Optional[List[str]] = None  # Teaching strategies from KG
    
    @property
    def is_lesson_intent(self) -> bool:
        """Check if this is a lesson/activity creation intent"""
        return self.query_intent in ("lesson_creation", "activity_design")
    
    @property
    def needs_external_apis(self) -> bool:
        """Check if this plan requires external API calls for subject content"""
        return self.scope_status in ("partial_scope", "out_of_scope")
    
    @property
    def is_in_scope(self) -> bool:
        """Check if query is fully within Knowledge Graph scope"""
        return self.scope_status == "in_scope"


class PlannerAgent:
    """
    Planner Agent - First step in the lesson planning pipeline.
    
    Responsibilities:
    1. Analyze the teacher's natural language query
    2. Identify key educational concepts to search
    3. Create a structured retrieval plan
    4. Determine lesson type, grade level, and constraints
    """
    
    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the Planner Agent.
        
        Args:
            model: OpenAI model to use for planning
        """
        self.model = model
        self._client: Optional[AsyncOpenAI] = None
    
    def _get_client(self) -> AsyncOpenAI:
        """Lazy initialization of OpenAI client"""
        if self._client is None:
            self._client = app_config.openai.get_async_client()
        return self._client
    
    async def plan(
        self,
        query: str,
        domain: str = "neuro",
        language: str = "it"
    ) -> RetrievalPlan:
        """
        Analyze a teacher query and create a retrieval plan.
        
        Args:
            query: Teacher's natural language query
            domain: Knowledge domain ("neuro" or "udl")
            language: Response language ("it" or "en")
            
        Returns:
            RetrievalPlan with structured search parameters
        """
        logger.info(f"[PlannerAgent] Analyzing query: {query[:50]}...")
        
        client = self._get_client()
        
        # Format the user prompt
        user_prompt = PLANNER_USER_TEMPLATE.format(
            query=query,
            domain=domain,
            language=language
        )
        
        try:
            completion_kwargs = app_config.openai.build_completion_kwargs(
                temperature=0.3,
                max_tokens=2000,
                json_mode=True,
            )
            response = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                **completion_kwargs
            )

            # Parse JSON response (extract_response_content also logs thinking tokens)
            content = extract_response_content(response, logger)
            plan_data = json.loads(content)
            
            # Extract query intent (with fallback for backward compatibility)
            query_intent = plan_data.get("query_intent", "lesson_creation")
            
            # NEW Phase A: Extract scope detection fields
            scope_status = plan_data.get("scope_status", "in_scope")
            scope_confidence = plan_data.get("scope_confidence", 1.0)
            subject_concepts = plan_data.get("subject_concepts")
            pedagogy_concepts = plan_data.get("pedagogy_concepts")
            
            plan = RetrievalPlan(
                query_intent=query_intent,
                key_concepts=plan_data.get("key_concepts", []),
                search_queries=plan_data.get("search_queries", [query]),
                lesson_type=plan_data.get("lesson_type"),  # Only set for lesson intents
                target_grade=plan_data.get("target_grade"),
                special_needs=plan_data.get("special_needs"),
                time_constraints=plan_data.get("time_constraints"),
                intent_confidence=plan_data.get("intent_confidence", "MEDIUM"),
                reasoning=plan_data.get("reasoning"),
                # NEW Phase A: Scope detection
                scope_status=scope_status,
                scope_confidence=scope_confidence,
                subject_concepts=subject_concepts,
                pedagogy_concepts=pedagogy_concepts
            )
            
            # Enhanced logging with scope status
            scope_emoji = {"in_scope": "✅", "partial_scope": "⚠️", "out_of_scope": "❌"}.get(scope_status, "❓")
            logger.info(
                f"[PlannerAgent] Created plan: intent={plan.query_intent}, "
                f"scope={scope_emoji} {scope_status} ({scope_confidence:.0%}), "
                f"{len(plan.search_queries)} queries, "
                f"concepts: {plan.key_concepts[:3]}..."
            )
            
            if plan.needs_external_apis:
                logger.info(f"[PlannerAgent] 🌐 External APIs needed for: {subject_concepts}")
            
            return plan
            
        except json.JSONDecodeError as e:
            logger.error(f"[PlannerAgent] Failed to parse JSON response: {e}")
            # Fallback plan - default to lesson_creation for backward compatibility
            return RetrievalPlan(
                query_intent="lesson_creation",
                key_concepts=[],
                search_queries=[query],
                lesson_type="full_lesson",
                intent_confidence="LOW",
                reasoning="Fallback plan due to parsing error"
            )
        except Exception as e:
            logger.error(f"[PlannerAgent] Planning failed: {e}")
            raise
    
    def plan_sync(
        self,
        query: str,
        domain: str = "neuro",
        language: str = "it"
    ) -> RetrievalPlan:
        """Synchronous version of plan()"""
        import asyncio
        return asyncio.run(self.plan(query, domain, language))

