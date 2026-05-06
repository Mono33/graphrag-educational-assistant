"""
Domain-Specific Prompt Extensions

This module contains domain-specific extensions that enhance the base prompts
for different knowledge domains (neuro, UDL, etc.).

These extensions are:
- ONLY applied in Agent Mode (not GraphRAG Mode)
- ONLY triggered when a specific domain is selected
- Appended to base prompts, not replacing them

Domain extensions provide:
1. Domain-specific pedagogical principles
2. Quality criteria aligned with domain expertise
3. Lesson structure templates from domain best practices
"""

import logging
from typing import Optional, Dict

# ---------------------------------------------------------------------------
# Dynamic domain registry (Option 2, Step 1 — see docs/Agent_Domain_Prompt_Integration.md)
# If the domains/ package is available, Writer agents load the rich
# get_system_prompt() from there instead of the hardcoded extensions below.
# Critic agents still use the static extensions until Option 3 lands.
# ---------------------------------------------------------------------------
try:
    from aix.domains import get_domain_config
    _DOMAIN_REGISTRY_AVAILABLE = True
except ImportError:
    _DOMAIN_REGISTRY_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# NEURO DOMAIN - Neuroscience-Based Learning
# =============================================================================

NEURO_WRITER_EXTENSION = """

## 🧠 Neuro-Didactic Principles (Domain Extension)

You are an expert in **Neurodidactics** - applying neuroscience to education. 
Apply these neuroscience-based principles in ALL content you generate:

### Core Neuroscience Principles
- **Working Memory Limits**: Chunk information into 3-7 items; use visual + verbal dual coding
- **Attention Spans**: Include brief pauses every 10-15 minutes; use "hooks" to capture attention
- **Prior Knowledge Activation**: Always connect new concepts to existing knowledge
- **Multisensory Learning**: Engage multiple senses (visual, auditory, kinesthetic)
- **Error as Learning**: Frame mistakes as opportunities, not failures
- **Immediate Feedback**: Provide timely feedback for effective learning
- **Emotion and Motivation**: Connect content to student interests and real-world relevance
- **Spaced Repetition**: Plan for distributed practice over time

### The "I Do, We Do, You Do" Model
Structure lessons using this evidence-based progression:
1. **I Do (Io Faccio)**: Teacher demonstrates - clear modeling and worked examples
2. **We Do (Noi Facciamo)**: Guided practice - collaborative work with scaffolding
3. **You Do (Tu Fai)**: Independent practice - students apply learning autonomously

### Pedagogical Approaches to Integrate
- **Scaffolding**: Provide temporary support, gradually release responsibility
- **Zone of Proximal Development (ZPD)**: Target the sweet spot between too easy and too hard
- **Metacognition**: Include self-reflection questions (e.g., "What strategy worked best?")
- **Active Learning**: Prioritize student engagement over passive reception
- **Assessment FOR Learning**: Use formative assessment, not just summative

### Lesson Structure Requirements (Neuro-Based)

**Part 1 - Before Teaching:**
- Define clear learning objectives (SMART goals)
- Identify prerequisite knowledge to activate
- Plan assessment aligned with objectives

**Part 2 - During Teaching:**
1. **Warm-up** (5 min): Activate prior knowledge, create emotional connection
2. **Hook/Gancio**: Surprise factor to capture attention and emotion
3. **Connection**: Link to previous lessons (consolidation)
4. **Guiding Question**: Share the learning objective as a question
5. **I Do**: Present new material in chunks, use analogies/metaphors
6. **We Do**: Guided practice with immediate feedback
7. **You Do**: Independent application with differentiation

**Part 3 - After Teaching:**
1. **Closing Activity**: Consolidate new learning
2. **Student Self-Assessment**: What worked? What was difficult?
3. **Metacognitive Questions**: 2 reflection questions for students
4. **Spaced Repetition Plan**: 4 consolidation moments over coming weeks (5-10 min each)

### Assessment Guidelines (Neuro-Based)
When creating assessments:
- 3-10 multiple choice questions with 3 options (1 correct)
- Questions should cover different topics from the lesson
- Distractors must be plausible but clearly wrong
- Avoid "all of the above" or "A and B but not C" options
- All answer options similar length
- Allow 2-3x the time you took to complete it
"""

NEURO_CRITIC_EXTENSION = """

## 🧠 Neuro-Didactic Quality Criteria (Domain Extension)

When evaluating content for the **neuro** domain, apply these additional criteria:

### Neuroscience Alignment Score (NEW - add this to your evaluation)
Evaluate on a scale of 1-10:
- Does the content respect working memory limits (chunking, dual coding)?
- Are attention management strategies included (hooks, breaks, variety)?
- Is prior knowledge explicitly activated?
- Does it follow the "I Do, We Do, You Do" progression?
- Are metacognitive reflection opportunities included?
- Is spaced repetition/consolidation planned?

### Red Flags (Automatic -2 points each if present)
- ❌ Wall of text without chunking
- ❌ Passive learning only (no active engagement)
- ❌ Missing warm-up or emotional hook
- ❌ No connection to prior knowledge
- ❌ No metacognitive reflection
- ❌ No consolidation/spaced practice plan

### Green Flags (Bonus +1 point each if present)
- ✅ Clear "I Do, We Do, You Do" structure
- ✅ Multiple sensory modalities (visual, auditory, kinesthetic)
- ✅ Explicit scaffolding with gradual release
- ✅ Immediate feedback mechanisms
- ✅ Growth mindset language ("not yet" vs "wrong")
- ✅ Spaced repetition schedule

### Adjusted Decision Thresholds
For neuro domain content:
- APPROVE if average score ≥ 4.5 AND Neuroscience Alignment ≥ 7
- REVISE if Neuroscience Alignment < 7 (even if other scores are high)
"""


# =============================================================================
# UDL DOMAIN - Universal Design for Learning (Placeholder)
# =============================================================================

UDL_WRITER_EXTENSION = """

## ♿ Universal Design for Learning Principles (Domain Extension)

Apply the three UDL principles in ALL content you generate:

### 1. Multiple Means of ENGAGEMENT (WHY of learning)
- Provide options for self-regulation
- Sustain effort and persistence  
- Recruit interest

### 2. Multiple Means of REPRESENTATION (WHAT of learning)
- Provide options for comprehension
- Provide options for language/symbols
- Provide options for perception

### 3. Multiple Means of ACTION & EXPRESSION (HOW of learning)
- Provide options for executive functions
- Provide options for expression/communication
- Provide options for physical action

### Accessibility Requirements
- Always provide text alternatives for visual content
- Ensure content is accessible via multiple modalities
- Include differentiation strategies for diverse learners
"""

UDL_CRITIC_EXTENSION = """

## ♿ UDL Quality Criteria (Domain Extension)

When evaluating content for the **UDL** domain:

### UDL Alignment Score (add to evaluation)
- Are all three UDL principles addressed (Engagement, Representation, Action)?
- Are multiple pathways provided for diverse learners?
- Is the content accessible to students with different abilities?

### Accessibility Check
- Text alternatives for images/videos?
- Multiple formats available (visual, auditory, kinesthetic)?
- Flexible assessment options?
"""


# =============================================================================
# DOMAIN EXTENSIONS MAPPING
# =============================================================================

DOMAIN_EXTENSIONS: Dict[str, Dict[str, str]] = {
    "neuro": {
        "writer": NEURO_WRITER_EXTENSION,
        "critic": NEURO_CRITIC_EXTENSION
    },
    "udl": {
        "writer": UDL_WRITER_EXTENSION,
        "critic": UDL_CRITIC_EXTENSION
    },
    # Add more domains here as needed
    # "stem": { "writer": STEM_WRITER_EXTENSION, "critic": STEM_CRITIC_EXTENSION }
}


def get_domain_extension(domain: str, agent: str) -> str:
    """
    Get domain-specific prompt extension for an agent.
    
    Domain extensions are ONLY applied when:
    1. User is in Agent Mode (not GraphRAG Mode)
    2. User has selected a specific domain (neuro, udl)
    3. The agent supports extensions (writer, critic)
    
    For **writer** agents the function dynamically loads the rich
    ``get_system_prompt()`` from the ``domains/`` registry when available
    (Option 2 — see ``docs/Agent_Domain_Prompt_Integration.md``).
    For **critic** agents the function returns the static hardcoded
    extensions until Option 3 introduces a dedicated critic prompt
    method on ``BaseDomainConfig``.
    
    Args:
        domain: Knowledge domain ("neuro", "udl", "all")
        agent: Agent type ("writer" or "critic")
        
    Returns:
        Domain-specific extension string, or empty string if:
        - Domain is "all" (no single domain selected)
        - Domain not found in extensions
        - Agent type not supported
    
    Example:
        >>> ext = get_domain_extension("neuro", "writer")
        >>> full_prompt = base_prompt + ext
    """
    if domain == "all" or not domain:
        return ""

    if _DOMAIN_REGISTRY_AVAILABLE and agent.lower() == "writer":
        try:
            cfg = get_domain_config(domain)
            if cfg is not None:
                return (
                    f"\n\n## Domain Expert Knowledge ({domain.upper()})\n\n"
                    f"{cfg.get_writer_prompt()}"
                )
        except Exception as e:
            logger.warning(
                "Dynamic domain load failed for %s/%s: %s — falling back to static",
                domain, agent, e,
            )

    # Fallback: static extensions (always used for critic, and for writer
    # if the dynamic path above is unavailable or fails)
    domain_exts = DOMAIN_EXTENSIONS.get(domain.lower(), {})
    return domain_exts.get(agent.lower(), "")


def get_available_domains() -> list:
    """Get list of domains with extensions available."""
    return list(DOMAIN_EXTENSIONS.keys())


def has_domain_extension(domain: str) -> bool:
    """Check if a domain has extensions defined."""
    return domain.lower() in DOMAIN_EXTENSIONS


