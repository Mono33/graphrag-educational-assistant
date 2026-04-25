"""
Agent Prompts

Specialized system prompts for each agent role.
These define the behavior and expertise of each agent.
"""

from aix.agent.prompts.planner_prompt import PLANNER_SYSTEM_PROMPT
from aix.agent.prompts.writer_prompt import WRITER_SYSTEM_PROMPT
from aix.agent.prompts.critic_prompt import CRITIC_SYSTEM_PROMPT

__all__ = [
    "PLANNER_SYSTEM_PROMPT",
    "WRITER_SYSTEM_PROMPT", 
    "CRITIC_SYSTEM_PROMPT"
]

