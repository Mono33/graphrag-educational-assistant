"""
Individual Agent Implementations

Each agent has a specialized role in the lesson planning pipeline:
- PlannerAgent: Analyzes teacher query and creates retrieval plan
- RetrieverAgent: Calls GraphRAG to fetch relevant knowledge
- WriterAgent: Generates the lesson plan from retrieved context
- CriticAgent: Reviews quality and requests revisions if needed
- GraphUpdaterAgent: (Phase 3) Updates knowledge graph with new content
"""

from aix.agent.agents.critic_agent import CriticAgent
from aix.agent.agents.planner_agent import PlannerAgent
from aix.agent.agents.retriever_agent import RetrieverAgent
from aix.agent.agents.writer_agent import WriterAgent

__all__ = ["PlannerAgent", "RetrieverAgent", "WriterAgent", "CriticAgent"]
