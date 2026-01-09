"""
Writer Agent

Generates educational content adapted to the detected query intent.
Supports lesson plans, definitions, comparisons, explanations, recommendations, and lists.
"""

import logging
from typing import Optional, Dict, Any

from openai import AsyncOpenAI

from agent.prompts.writer_prompt import (
    get_writer_prompts,
    WRITER_REVISION_TEMPLATE,
    WRITER_USER_TEMPLATE_LESSON,
    WRITER_USER_TEMPLATE_DEFINITION,
    WRITER_USER_TEMPLATE_COMPARISON,
    WRITER_USER_TEMPLATE_EXPLANATION,
    WRITER_USER_TEMPLATE_RECOMMENDATION,
    WRITER_USER_TEMPLATE_LIST,
)
from agent.agents.planner_agent import RetrievalPlan
from agent.agents.retriever_agent import RetrievalResult

logger = logging.getLogger(__name__)


class WriterAgent:
    """
    Writer Agent - Generates educational content adapted to query intent.
    
    Responsibilities:
    1. Detect query intent from the plan
    2. Select appropriate output format and template
    3. Generate content grounded in retrieved evidence
    4. Handle revision requests from CriticAgent
    """
    
    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the Writer Agent.
        
        Args:
            model: OpenAI model to use for generation
        """
        self.model = model
        self._client: Optional[AsyncOpenAI] = None
    
    def _get_client(self) -> AsyncOpenAI:
        """Lazy initialization of OpenAI client"""
        if self._client is None:
            self._client = AsyncOpenAI()
        return self._client
    
    async def write(
        self,
        teacher_query: str,
        plan: RetrievalPlan,
        retrieval_result: RetrievalResult,
        language: str = "it"
    ) -> str:
        """
        Generate content based on the detected intent and retrieved context.
        
        Args:
            teacher_query: Original teacher request
            plan: Retrieval plan with intent and lesson requirements
            retrieval_result: Retrieved knowledge from GraphRAG
            language: Output language ("it" or "en")
            
        Returns:
            Generated content as markdown string
        """
        # Get the query intent from the plan
        intent = getattr(plan, 'query_intent', 'lesson_creation')
        
        logger.info(f"[WriterAgent] Generating content for intent: {intent}")
        
        client = self._get_client()
        
        # Get intent-specific prompts
        system_prompt, user_template = get_writer_prompts(intent)
        
        # Format retrieved nodes for prompt
        nodes_text = self._format_nodes(retrieval_result.nodes)
        recommendations_text = self._format_recommendations(retrieval_result.recommendations)
        key_concepts_text = ', '.join(plan.key_concepts) if plan.key_concepts else "Not specified"
        
        # Format user prompt based on intent
        if intent in ("lesson_creation", "activity_design"):
            user_prompt = user_template.format(
                teacher_query=teacher_query,
                key_concepts=key_concepts_text,
                recommendations=recommendations_text,
                retrieved_nodes=nodes_text,
                lesson_type=plan.lesson_type or "full_lesson",
                target_grade=plan.target_grade or "Not specified",
                time_constraints=plan.time_constraints or "Not specified",
                special_needs=', '.join(plan.special_needs) if plan.special_needs else "None",
                language="Italian" if language == "it" else "English"
            )
        elif intent == "recommendation":
            user_prompt = WRITER_USER_TEMPLATE_RECOMMENDATION.format(
                teacher_query=teacher_query,
                key_concepts=key_concepts_text,
                recommendations=recommendations_text,
                retrieved_nodes=nodes_text,
                special_needs=', '.join(plan.special_needs) if plan.special_needs else "None",
                language="Italian" if language == "it" else "English"
            )
        else:
            # definition, comparison, explanation, list
            user_prompt = user_template.format(
                teacher_query=teacher_query,
                key_concepts=key_concepts_text,
                retrieved_nodes=nodes_text,
                language="Italian" if language == "it" else "English"
            )
        
        # Format system prompt with language
        system_prompt = system_prompt.replace(
            "{language}",
            "Italian" if language == "it" else "English"
        )
        
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7  # Balanced creativity and consistency
            )
            
            content = response.choices[0].message.content
            
            logger.info(
                f"[WriterAgent] Generated {intent} content "
                f"({len(content)} characters)"
            )
            
            return content
            
        except Exception as e:
            logger.error(f"[WriterAgent] Generation failed: {e}")
            raise
    
    async def revise(
        self,
        current_draft: str,
        critique: str,
        revision_instructions: str,
        language: str = "it",
        intent: str = "lesson_creation"
    ) -> str:
        """
        Revise content based on critic feedback.
        
        Args:
            current_draft: Current content draft
            critique: Critic's feedback
            revision_instructions: Specific revision instructions
            language: Output language
            intent: Query intent for selecting the right system prompt
            
        Returns:
            Revised content
        """
        logger.info(f"[WriterAgent] Revising {intent} content based on feedback...")
        
        client = self._get_client()
        
        user_prompt = WRITER_REVISION_TEMPLATE.format(
            current_draft=current_draft,
            critique=critique,
            revision_instructions=revision_instructions
        )
        
        # Get intent-specific system prompt for consistent formatting
        system_prompt, _ = get_writer_prompts(intent)
        system_prompt = system_prompt.replace(
            "{language}",
            "Italian" if language == "it" else "English"
        )
        
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5  # Lower for revisions
            )
            
            revised_content = response.choices[0].message.content
            
            logger.info(
                f"[WriterAgent] Revised content "
                f"({len(revised_content)} characters)"
            )
            
            return revised_content
            
        except Exception as e:
            logger.error(f"[WriterAgent] Revision failed: {e}")
            raise
    
    def _format_nodes(self, nodes: list, max_nodes: int = 20) -> str:
        """Format retrieved nodes for the prompt"""
        if not nodes:
            return "No nodes retrieved"
        
        lines = []
        for node in nodes[:max_nodes]:
            name = node.get('name', 'Unknown')
            labels = node.get('labels', [])
            desc = node.get('description', '')[:150]
            
            label_str = f" [{', '.join(labels)}]" if labels else ""
            lines.append(f"- **{name}**{label_str}")
            if desc:
                lines.append(f"  {desc}")
        
        return '\n'.join(lines)
    
    def _format_recommendations(self, recommendations: list, max_recs: int = 10) -> str:
        """Format recommendations for the prompt"""
        if not recommendations:
            return "No specific recommendations"
        
        lines = []
        for rec in recommendations[:max_recs]:
            name = rec.get('name', 'Unknown')
            confidence = rec.get('confidence', 'MEDIUM')
            desc = rec.get('description', '')[:200]
            
            lines.append(f"### {name} (Confidence: {confidence})")
            if desc:
                lines.append(desc)
            lines.append("")
        
        return '\n'.join(lines)
    
    def write_sync(
        self,
        teacher_query: str,
        plan: RetrievalPlan,
        retrieval_result: RetrievalResult,
        language: str = "it"
    ) -> str:
        """Synchronous version of write()"""
        import asyncio
        return asyncio.run(self.write(teacher_query, plan, retrieval_result, language))
