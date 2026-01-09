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
    
    @property
    def is_lesson_intent(self) -> bool:
        """Check if this is a lesson/activity creation intent"""
        return self.query_intent in ("lesson_creation", "activity_design")


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
            self._client = AsyncOpenAI()
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
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3  # Lower temperature for consistent planning
            )
            
            # Parse JSON response
            content = response.choices[0].message.content
            plan_data = json.loads(content)
            
            # Extract query intent (with fallback for backward compatibility)
            query_intent = plan_data.get("query_intent", "lesson_creation")
            
            plan = RetrievalPlan(
                query_intent=query_intent,
                key_concepts=plan_data.get("key_concepts", []),
                search_queries=plan_data.get("search_queries", [query]),
                lesson_type=plan_data.get("lesson_type"),  # Only set for lesson intents
                target_grade=plan_data.get("target_grade"),
                special_needs=plan_data.get("special_needs"),
                time_constraints=plan_data.get("time_constraints"),
                intent_confidence=plan_data.get("intent_confidence", "MEDIUM"),
                reasoning=plan_data.get("reasoning")
            )
            
            logger.info(
                f"[PlannerAgent] Created plan: intent={plan.query_intent}, "
                f"confidence={plan.intent_confidence}, "
                f"{len(plan.search_queries)} queries, "
                f"concepts: {plan.key_concepts[:3]}..."
            )
            
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

