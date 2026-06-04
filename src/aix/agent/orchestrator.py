"""
Agent Orchestrator - Main Entry Point

This is the primary interface for using the multi-agent lesson planning system.
It provides a clean API that hides the underlying LangGraph complexity.
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

# Load environment variables FIRST (for OPENAI_API_KEY)
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class LessonPlanResult:
    """Result from the lesson planning pipeline"""

    success: bool
    lesson_plan: Optional[str]
    approved: bool
    revision_count: int
    scores: Optional[dict[str, int]]
    nodes_used: int
    recommendations_used: int
    critique_summary: Optional[str]
    error: Optional[str] = None
    # Phase 3: Added for upsell buttons
    query_intent: Optional[str] = "lesson_creation"
    key_concepts: Optional[list[str]] = None
    # Phase 3 (Media): Curated media from sidecar JSON
    curated_media: Optional[dict[str, Any]] = None
    # Phase A: Scope detection for hybrid mode
    scope_status: Optional[str] = "in_scope"  # in_scope, partial_scope, out_of_scope
    scope_confidence: Optional[float] = 1.0
    external_resources: Optional[dict[str, Any]] = None  # Wikipedia, papers, OER

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "lesson_plan": self.lesson_plan,
            "approved": self.approved,
            "revision_count": self.revision_count,
            "scores": self.scores,
            "nodes_used": self.nodes_used,
            "recommendations_used": self.recommendations_used,
            "critique_summary": self.critique_summary,
            "error": self.error,
            "query_intent": self.query_intent,
            "key_concepts": self.key_concepts,
            "curated_media": self.curated_media,
            "scope_status": self.scope_status,
            "scope_confidence": self.scope_confidence,
            "external_resources": self.external_resources,
        }

    @property
    def has_media(self) -> bool:
        """Check if curated media is available"""
        return bool(self.curated_media and any(self.curated_media.values()))

    @property
    def is_hybrid(self) -> bool:
        """Check if this is a hybrid result (external + KG sources)"""
        return self.scope_status in ("partial_scope", "out_of_scope")

    @property
    def has_external_resources(self) -> bool:
        """Check if external resources were used"""
        return bool(self.external_resources and any(self.external_resources.values()))


class AgentOrchestrator:
    """
    Main orchestrator for the Agentic GraphRAG system.

    This class provides a simple, clean API for creating lesson plans
    using the multi-agent pipeline (Planner → Retriever → Writer → Critic).

    Usage:
        # Async usage
        orchestrator = AgentOrchestrator(domain="neuro")
        result = await orchestrator.create_lesson_plan(
            "Crea una lezione sulla motivazione per studenti con ADHD"
        )
        print(result.lesson_plan)

        # Sync usage
        result = orchestrator.create_lesson_plan_sync(
            "Crea una lezione sulla metacognizione"
        )

    The orchestrator:
    1. Uses your EXISTING GraphRAG engine (no modifications)
    2. Adds planning, writing, and quality review capabilities
    3. Supports automatic revision cycles
    4. Works with both "neuro" and "udl" domains
    """

    def __init__(
        self,
        domain: str = "neuro",
        language: str = "it",
        max_revisions: int = 2,
        model: str = "gpt-4o",
    ):
        """
        Initialize the Agent Orchestrator.

        Args:
            domain: Knowledge domain
                - "neuro": Neuroscience-based learning strategies
                - "udl": Universal Design for Learning
            language: Output language
                - "it": Italian
                - "en": English
            max_revisions: Maximum revision cycles before auto-approve
            model: OpenAI model for all agents
        """
        self.domain = domain
        self.language = language
        self.max_revisions = max_revisions
        self.model = model

        self._pipeline = None

        logger.info(
            f"[Orchestrator] Initialized for domain='{domain}', "
            f"language='{language}', max_revisions={max_revisions}"
        )

    def _get_pipeline(self):
        """Lazy initialization of the pipeline"""
        if self._pipeline is None:
            from aix.agent.graph.lesson_planner_graph import LessonPlannerPipeline

            self._pipeline = LessonPlannerPipeline(
                domain=self.domain, language=self.language, max_revisions=self.max_revisions
            )
        return self._pipeline

    async def create_lesson_plan(
        self,
        query: str,
        session_id: Optional[str] = None,
        educational_profile: Optional[Any] = None,
    ) -> LessonPlanResult:
        """
        Create a lesson plan from a teacher's query.

        This runs the full multi-agent pipeline:
        1. Planner analyzes the query
        2. Retriever searches the knowledge graph
        3. Writer generates the lesson plan
        4. Critic reviews and may request revisions

        Args:
            query: Teacher's natural language query
                Examples:
                - "Crea una lezione sulla motivazione per studenti con ADHD"
                - "Attività di 30 minuti sulla metacognizione"
                - "How to teach executive functions to high school students"
            session_id: Optional session ID for tracking/persistence
            educational_profile: Optional per-request class/classroom context
                (CORE 1 #2.5). Either a Pydantic `EducationalProfile` or a
                pre-serialized `dict`. Backward compatible — when omitted,
                the pipeline keeps the original generic behavior.

        Returns:
            LessonPlanResult containing the lesson plan and metadata
        """
        logger.info(f"[Orchestrator] Creating lesson plan: {query[:50]}...")

        pipeline = self._get_pipeline()

        try:
            result = await pipeline.run(query, session_id, educational_profile=educational_profile)

            return LessonPlanResult(
                success=result.get("success", False),
                lesson_plan=result.get("lesson_plan"),
                approved=result.get("approved", False),
                revision_count=result.get("revision_count", 0),
                scores=result.get("scores"),
                nodes_used=result.get("sources", {}).get("nodes_count", 0),
                recommendations_used=result.get("sources", {}).get("recommendations_count", 0),
                critique_summary=result.get("critique"),
                error=result.get("error"),
                # Phase 3: Pass query_intent and key_concepts for upsell buttons
                query_intent=result.get("query_intent", "lesson_creation"),
                key_concepts=result.get("key_concepts", []),
                # Phase 3 (Media): Pass curated media for enhancement buttons
                curated_media=result.get("curated_media"),
                # Phase A: Scope detection for hybrid mode
                scope_status=result.get("scope_status", "in_scope"),
                scope_confidence=result.get("scope_confidence", 1.0),
                external_resources=result.get("external_resources"),
            )

        except Exception as e:
            logger.error(f"[Orchestrator] Pipeline failed: {e}")
            return LessonPlanResult(
                success=False,
                lesson_plan=None,
                approved=False,
                revision_count=0,
                scores=None,
                nodes_used=0,
                recommendations_used=0,
                critique_summary=None,
                error=str(e),
            )

    def create_lesson_plan_sync(
        self,
        query: str,
        session_id: Optional[str] = None,
        educational_profile: Optional[Any] = None,
    ) -> LessonPlanResult:
        """
        Synchronous version of create_lesson_plan().

        Use this in non-async contexts (e.g., scripts, Jupyter notebooks).
        """
        import asyncio

        return asyncio.run(
            self.create_lesson_plan(query, session_id, educational_profile=educational_profile)
        )

    async def quick_search(self, query: str) -> dict[str, Any]:
        """
        Quick search without lesson plan generation.

        Useful for exploring the knowledge graph or testing retrieval.

        Args:
            query: Search query

        Returns:
            Dictionary with retrieved nodes and recommendations
        """
        from aix.agent.tools.graphrag_tool import GraphRAGTool

        tool = GraphRAGTool(domain=self.domain)
        result = await tool.search(query)

        return {
            "nodes": result.nodes,
            "relationships": result.relationships,
            "recommendations": result.recommendations,
            "confidence": result.confidence,
        }

    def quick_search_sync(self, query: str) -> dict[str, Any]:
        """Synchronous version of quick_search()"""
        import asyncio

        return asyncio.run(self.quick_search(query))


# CLI Interface
def main():
    """
    CLI interface for testing the orchestrator.

    Usage:
        python -m agent.orchestrator "Your query here"
    """
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Create lesson plans using Agentic GraphRAG")
    parser.add_argument("query", type=str, help="Teacher's query in natural language")
    parser.add_argument(
        "--domain", type=str, default="neuro", choices=["neuro", "udl"], help="Knowledge domain"
    )
    parser.add_argument(
        "--language", type=str, default="it", choices=["it", "en"], help="Output language"
    )
    parser.add_argument("--max-revisions", type=int, default=2, help="Maximum revision cycles")
    parser.add_argument("--output", type=str, help="Output file path (optional)")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("\n🎓 Agentic GraphRAG Lesson Planner")
    print("=" * 50)
    print(f"Query: {args.query}")
    print(f"Domain: {args.domain}")
    print(f"Language: {args.language}")
    print("=" * 50)

    # Create orchestrator
    orchestrator = AgentOrchestrator(
        domain=args.domain, language=args.language, max_revisions=args.max_revisions
    )

    # Run pipeline
    result = asyncio.run(orchestrator.create_lesson_plan(args.query))

    # Output results
    print("\n📊 Results")
    print("-" * 50)
    print(f"Success: {result.success}")
    print(f"Approved: {result.approved}")
    print(f"Revisions: {result.revision_count}")
    print(f"Nodes Used: {result.nodes_used}")
    print(f"Recommendations: {result.recommendations_used}")

    if result.scores:
        print(f"\nScores: {result.scores}")

    if result.error:
        print(f"\n❌ Error: {result.error}")

    if result.lesson_plan:
        print("\n📝 Lesson Plan")
        print("=" * 50)
        print(result.lesson_plan)

        # Save to file if specified
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(result.lesson_plan)
            print(f"\n✅ Saved to: {args.output}")


if __name__ == "__main__":
    main()
