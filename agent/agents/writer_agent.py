"""
Writer Agent

Generates educational content adapted to the detected query intent.
Supports lesson plans, definitions, comparisons, explanations, recommendations, and lists.

Enhanced with curated media integration (Phase 2) for multimodal content support.
"""

import logging
from typing import Optional, Dict, Any, List

from openai import AsyncOpenAI
from config import config as app_config, extract_response_content

from agent.prompts.writer_prompt import (
    get_writer_prompts,
    format_external_resources,
    WRITER_REVISION_TEMPLATE,
    WRITER_USER_TEMPLATE_LESSON,
    WRITER_USER_TEMPLATE_DEFINITION,
    WRITER_USER_TEMPLATE_COMPARISON,
    WRITER_USER_TEMPLATE_EXPLANATION,
    WRITER_USER_TEMPLATE_RECOMMENDATION,
    WRITER_USER_TEMPLATE_LIST,
    WRITER_USER_TEMPLATE_HYBRID,
)
from agent.agents.planner_agent import RetrievalPlan
from agent.agents.retriever_agent import RetrievalResult

# Optional domain extensions - fails gracefully if not available
try:
    from agent.configs.domain_prompts import get_domain_extension
    DOMAIN_EXTENSIONS_AVAILABLE = True
except ImportError:
    DOMAIN_EXTENSIONS_AVAILABLE = False
    get_domain_extension = lambda d, a: ""  # Fallback: no extension

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
            self._client = app_config.openai.get_async_client()
        return self._client
    
    async def write(
        self,
        teacher_query: str,
        plan: RetrievalPlan,
        retrieval_result: RetrievalResult,
        language: str = "it",
        curated_media: Optional[Dict[str, Any]] = None,  # Phase 2: Optional media
        external_resources: Optional[Dict[str, Any]] = None,  # Phase A: External resources
        domain: str = "neuro"  # Phase B: Domain for extensions
    ) -> str:
        """
        Generate content based on the detected intent and retrieved context.
        
        Args:
            teacher_query: Original teacher request
            plan: Retrieval plan with intent and lesson requirements
            retrieval_result: Retrieved knowledge from GraphRAG
            language: Output language ("it" or "en")
            curated_media: Optional curated media from sidecar JSON (Phase 2)
            external_resources: Optional external resources for hybrid mode (Phase A)
            
        Returns:
            Generated content as markdown string
        """
        # Get the query intent and scope from the plan
        intent = getattr(plan, 'query_intent', 'lesson_creation')
        scope_status = getattr(plan, 'scope_status', 'in_scope')
        
        # Check for media/resources availability
        has_media = bool(curated_media and any(curated_media.values()))
        has_external = bool(external_resources and any(external_resources.values()))
        
        # Determine if this is hybrid mode
        is_hybrid = scope_status in ("partial_scope", "out_of_scope") and has_external
        
        scope_emoji = {"in_scope": "✅", "partial_scope": "⚠️", "out_of_scope": "❌"}.get(scope_status, "❓")
        logger.info(
            f"[WriterAgent] Generating content: intent={intent}, scope={scope_emoji} {scope_status}, "
            f"media={has_media}, hybrid={is_hybrid}"
        )
        
        client = self._get_client()
        
        # Get intent-specific prompts (with scope awareness for hybrid mode)
        system_prompt, user_template = get_writer_prompts(intent, scope_status)
        
        # Format retrieved nodes for prompt
        nodes_text = self._format_nodes(retrieval_result.nodes)
        recommendations_text = self._format_recommendations(retrieval_result.recommendations)
        key_concepts_text = ', '.join(plan.key_concepts) if plan.key_concepts else "Not specified"
        
        # NEW Phase 2: Format curated media if available
        media_text = self._format_media(curated_media) if has_media else ""
        
        # NEW Phase A: Handle HYBRID mode (out-of-scope with external resources)
        if is_hybrid and intent in ("lesson_creation", "activity_design"):
            wikipedia_content, papers_content, oer_content = format_external_resources(external_resources)
            
            user_prompt = WRITER_USER_TEMPLATE_HYBRID.format(
                teacher_query=teacher_query,
                wikipedia_content=wikipedia_content,
                papers_content=papers_content,
                oer_content=oer_content,
                recommendations=recommendations_text,
                retrieved_nodes=nodes_text,
                lesson_type=plan.lesson_type or "full_lesson",
                target_grade=plan.target_grade or "Not specified",
                time_constraints=plan.time_constraints or "Not specified",
                language="Italian" if language == "it" else "English"
            )
            
            logger.info(f"[WriterAgent] Using HYBRID mode prompt with external resources (OER: {len(external_resources.get('oer_textbooks', []))} textbooks)")
        
        # Standard mode: Format user prompt based on intent
        elif intent in ("lesson_creation", "activity_design"):
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
        
        # NEW Phase 2: Append media context if available
        if media_text:
            user_prompt += media_text
        
        # Format system prompt with language
        system_prompt = system_prompt.replace(
            "{language}",
            "Italian" if language == "it" else "English"
        )
        
        # NEW Phase B: Add domain-specific extensions
        if DOMAIN_EXTENSIONS_AVAILABLE:
            domain_ext = get_domain_extension(domain, "writer")
            if domain_ext:
                system_prompt += domain_ext
                logger.info(f"[WriterAgent] Applied domain extension for '{domain}'")
        
        try:
            completion_kwargs = app_config.openai.build_completion_kwargs(
                temperature=0.7,
                max_tokens=4000,
            )
            response = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                **completion_kwargs
            )

            content = extract_response_content(response, logger)
            
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
            completion_kwargs = app_config.openai.build_completion_kwargs(
                temperature=0.5,
                max_tokens=4000,
            )
            response = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                **completion_kwargs
            )

            revised_content = extract_response_content(response, logger)
            
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
    
    def _format_media(self, curated_media: Optional[Dict[str, Any]]) -> str:
        """
        Format curated media for inclusion in the prompt (Phase 2).
        
        This provides the LLM with available media resources to reference
        in the generated content.
        
        Args:
            curated_media: Dict with videos, resources, citations
            
        Returns:
            Formatted media context string (empty if no media)
        """
        if not curated_media:
            return ""
        
        lines = [
            "\n\n## 📚 Available Educational Media",
            "Use these curated resources to enrich your response. Include relevant links and citations."
        ]
        
        # Videos
        videos = curated_media.get('videos', [])
        if videos:
            lines.append("\n### 🎥 Video Educativi Disponibili")
            for v in videos[:5]:
                title = v.get('title', 'Video')
                url = v.get('url')
                search_query = v.get('search_query', '')
                duration = v.get('duration_hint', '')
                
                if url:
                    lines.append(f"- [{title}]({url})" + (f" ({duration})" if duration else ""))
                else:
                    lines.append(f"- **{title}** - Cerca su YouTube: \"{search_query}\"")
        
        # Images/Diagrams (descriptions for potential generation)
        images = curated_media.get('images', [])
        if images:
            lines.append("\n### 🖼️ Diagrammi/Immagini Suggeriti")
            lines.append("Puoi suggerire questi diagrammi nel contenuto:")
            for img in images[:3]:
                desc = img.get('description', '')
                img_type = img.get('type', 'diagram')
                lines.append(f"- [{img_type.title()}] {desc}")
        
        # External Resources
        resources = curated_media.get('resources', [])
        if resources:
            lines.append("\n### 🔗 Risorse Educative")
            for r in resources[:5]:
                title = r.get('title', 'Resource')
                url = r.get('url') or r.get('suggested_url')
                res_type = r.get('type', 'educational')
                
                if url:
                    lines.append(f"- [{title}]({url}) ({res_type})")
                else:
                    lines.append(f"- {title} ({res_type})")
        
        # Academic Citations
        citations = curated_media.get('citations', [])
        if citations:
            lines.append("\n### 📖 Riferimenti Scientifici")
            lines.append("Includi questi riferimenti nella sezione Fonti:")
            for c in citations[:3]:
                authors = c.get('authors', [])
                authors_str = ', '.join(authors[:2])
                if len(authors) > 2:
                    authors_str += ' et al.'
                year = c.get('year', '')
                title = c.get('title', '')
                journal = c.get('journal', '')
                doi = c.get('doi')
                
                cite_line = f"- {authors_str}"
                if year:
                    cite_line += f" ({year})"
                cite_line += f". *{title}*"
                if journal:
                    cite_line += f". {journal}"
                if doi:
                    cite_line += f" [DOI: {doi}]"
                lines.append(cite_line)
        
        # Instruction for the LLM
        lines.append("\n**Istruzioni:** Incorpora i link video e risorse pertinenti nel contenuto. ")
        lines.append("Cita i riferimenti scientifici nella sezione Fonti alla fine.")
        
        return '\n'.join(lines)
    
    def write_sync(
        self,
        teacher_query: str,
        plan: RetrievalPlan,
        retrieval_result: RetrievalResult,
        language: str = "it",
        curated_media: Optional[Dict[str, Any]] = None,
        external_resources: Optional[Dict[str, Any]] = None,
        domain: str = "neuro"
    ) -> str:
        """Synchronous version of write()"""
        import asyncio
        return asyncio.run(self.write(
            teacher_query, plan, retrieval_result, language, 
            curated_media, external_resources, domain
        ))
