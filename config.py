#!/usr/bin/env python3
"""
config.py - Configuration settings for the GraphRAG text2cypher module.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Neo4jConfig:
    """Neo4j database configuration"""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = ""
    database: str = "neo4j"
    encrypted: bool = True


@dataclass
class OpenAIConfig:
    """OpenAI-compatible API configuration (supports OpenRouter and OpenAI)"""
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "openai/gpt-4o"  # OpenRouter model ID format
    temperature: float = 0.1
    max_tokens: int = 500

    def get_client(self):
        """Return a configured OpenAI-compatible client (sync)"""
        from openai import OpenAI
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_async_client(self):
        """Return a configured OpenAI-compatible client (async)"""
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def is_reasoning_model(self) -> bool:
        """True for models that use internal chain-of-thought / thinking tokens.

        Reasoning models (o1, o4-mini, DeepSeek R1, Claude with thinking, etc.)
        have different API constraints:
        - Don't accept temperature < 1  (o1/o4-mini) or ignore it (DeepSeek R1)
        - Use max_completion_tokens instead of max_tokens  (o1/o4-mini)
        - Return thinking content in reasoning_content field  (DeepSeek R1, Claude via OpenRouter)
        """
        m = self.model.lower()
        return any(x in m for x in ("o1", "o3", "o4", "deepseek-r1", "deepseek/deepseek-r1", "thinking"))

    def build_completion_kwargs(
        self,
        *,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        json_mode: bool = False,
        include_reasoning: bool = True,
    ) -> dict:
        """Build chat.completions.create() kwargs compatible with the active model.

        Handles three families transparently:
        - Standard models (GPT-4o, Claude, Gemini): pass temperature + max_tokens
        - OpenAI o-series (o1, o3, o4-mini): drop temperature, use max_completion_tokens
        - DeepSeek R1 / Claude-thinking via OpenRouter: add include_reasoning in extra_body

        Args:
            temperature: Desired sampling temperature (ignored for o-series).
            max_tokens: Max output tokens.
            json_mode: Request JSON output via response_format (skipped for reasoning models).
            include_reasoning: Ask OpenRouter to return thinking tokens in reasoning_content.
        """
        kwargs: dict = {"model": self.model}
        is_o_series = any(x in self.model.lower() for x in ("o1", "o3", "o4"))
        is_thinking = self.is_reasoning_model()

        if is_o_series:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature

        if json_mode and not is_thinking:
            kwargs["response_format"] = {"type": "json_object"}

        if is_thinking and include_reasoning:
            kwargs["extra_body"] = {"include_reasoning": True}

        return kwargs


@dataclass
class Text2CypherConfig:
    """Text2Cypher module configuration"""
    max_query_length: int = 1000
    default_limit: int = 20
    enable_query_validation: bool = True
    enable_query_execution: bool = True
    log_level: str = "INFO"
    model: str = "google/gemini-2.0-flash"  # Fast model for Cypher generation and translation


@dataclass
class EmbeddingConfig:
    """Embedding mode configuration for hybrid retrieval

    Modes:
        - "node2vec": Graph structure only (default, backward compatible)
        - "hybrid_semantic": Node2Vec + OpenAI text embeddings
        - "openai_only": OpenAI embeddings only (no graph structure)

    Weight α (NODE2VEC_WEIGHT):
        - Controls balance between structure and semantics
        - α = 1.0: 100% Node2Vec (pure structure)
        - α = 0.0: 100% OpenAI semantic
        - α = 0.4: Recommended for educational queries (slight semantic bias)

    Why α = 0.4 (40% Node2Vec, 60% Semantic)?
        - Educational queries are often semantic ("what is X?", "difference between A and B")
        - Italian queries benefit from multilingual semantic embeddings
        - Graph structure still matters for finding connected concepts
        - Research shows 40/60 split optimal for Q&A over knowledge graphs
    """
    mode: str = "node2vec"  # "node2vec" | "hybrid_semantic" | "openai_only"

    # Embedding model — OpenRouter format (provider/model)
    embedding_model: str = "openai/text-embedding-3-small"

    # Hybrid weights (α for Node2Vec, 1-α for semantic)
    # α = 0.4 gives 40% weight to graph structure, 60% to semantic meaning
    node2vec_weight: float = 0.4  # α

    # Semantic similarity threshold (minimum score to consider)
    semantic_threshold: float = 0.3

    # Cache settings
    cache_embeddings: bool = True
    embeddings_cache_dir: str = "models/embeddings_cache"


class Config:
    """Main configuration class"""

    def __init__(self):
        self.neo4j = Neo4jConfig()
        self.openai = OpenAIConfig()
        self.text2cypher = Text2CypherConfig()
        self.embedding = EmbeddingConfig()
        self._load_from_env()

    def _load_from_env(self):
        """Load configuration from environment variables"""
        # Neo4j configuration
        self.neo4j.uri = os.getenv("NEO4J_URI", self.neo4j.uri)
        self.neo4j.user = os.getenv("NEO4J_USER", self.neo4j.user)
        self.neo4j.password = os.getenv("NEO4J_PASSWORD", self.neo4j.password)
        self.neo4j.database = os.getenv("NEO4J_DATABASE", self.neo4j.database)
        # Handle encrypted flag from env (1/0 or true/false)
        env_encrypted = os.getenv("NEO4J_ENCRYPTED", "1")
        self.neo4j.encrypted = env_encrypted == "1"

        # OpenRouter / OpenAI-compatible configuration
        # Prefer OPENROUTER_API_KEY; fall back to OPENAI_API_KEY for backward compat
        self.openai.api_key = (
            os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENAI_API_KEY", self.openai.api_key)
        )
        self.openai.base_url = os.getenv("OPENROUTER_BASE_URL", self.openai.base_url)
        self.openai.model = os.getenv("LLM_MODEL", self.openai.model)

        # Text2Cypher configuration
        if os.getenv("TEXT2CYPHER_MAX_QUERY_LENGTH"):
            self.text2cypher.max_query_length = int(os.getenv("TEXT2CYPHER_MAX_QUERY_LENGTH"))

        if os.getenv("TEXT2CYPHER_DEFAULT_LIMIT"):
            self.text2cypher.default_limit = int(os.getenv("TEXT2CYPHER_DEFAULT_LIMIT"))

        if os.getenv("TEXT2CYPHER_ENABLE_VALIDATION"):
            self.text2cypher.enable_query_validation = os.getenv("TEXT2CYPHER_ENABLE_VALIDATION").lower() == "true"

        if os.getenv("TEXT2CYPHER_ENABLE_EXECUTION"):
            self.text2cypher.enable_query_execution = os.getenv("TEXT2CYPHER_ENABLE_EXECUTION").lower() == "true"

        self.text2cypher.log_level = os.getenv("LOG_LEVEL", self.text2cypher.log_level)
        self.text2cypher.model = (
            os.getenv("TEXT2CYPHER_MODEL")
            or os.getenv("LLM_MODEL", self.text2cypher.model)
        )

        # Embedding configuration
        self.embedding.mode = os.getenv("EMBEDDING_MODE", self.embedding.mode)
        # Read EMBEDDING_MODEL first; fall back to old OPENAI_EMBEDDING_MODEL for backward compat
        self.embedding.embedding_model = (
            os.getenv("EMBEDDING_MODEL")
            or os.getenv("OPENAI_EMBEDDING_MODEL", self.embedding.embedding_model)
        )

        if os.getenv("EMBEDDING_NODE2VEC_WEIGHT"):
            self.embedding.node2vec_weight = float(os.getenv("EMBEDDING_NODE2VEC_WEIGHT"))

        if os.getenv("EMBEDDING_SEMANTIC_THRESHOLD"):
            self.embedding.semantic_threshold = float(os.getenv("EMBEDDING_SEMANTIC_THRESHOLD"))

        if os.getenv("EMBEDDING_CACHE_EMBEDDINGS"):
            self.embedding.cache_embeddings = os.getenv("EMBEDDING_CACHE_EMBEDDINGS").lower() == "true"

        self.embedding.embeddings_cache_dir = os.getenv(
            "EMBEDDINGS_CACHE_DIR",
            self.embedding.embeddings_cache_dir
        )

    def validate(self) -> tuple[bool, list[str]]:
        """Validate configuration and return any errors"""
        errors = []

        if not self.neo4j.password:
            errors.append("Neo4j password is required")

        if not self.openai.api_key:
            errors.append("OpenAI API key is required")

        if self.openai.temperature < 0 or self.openai.temperature > 1:
            errors.append("OpenAI temperature must be between 0 and 1")

        if self.text2cypher.max_query_length <= 0:
            errors.append("Max query length must be positive")

        if self.text2cypher.default_limit <= 0:
            errors.append("Default limit must be positive")

        return len(errors) == 0, errors


# Global configuration instance
config = Config()


def extract_response_content(response, logger=None) -> str:
    """Extract text content from a chat completion response.

    Also logs thinking tokens (reasoning_content) when present — useful for
    debugging reasoning models like DeepSeek R1 or Claude with extended thinking.

    Args:
        response: A chat.completions response object.
        logger: Optional logger; if None reasoning tokens are printed to stdout.

    Returns:
        The assistant's final text content.
    """
    import logging as _logging
    _log = logger or _logging.getLogger(__name__)

    message = response.choices[0].message
    content = message.content or ""

    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        _log.debug(f"[ThinkingTokens] {len(reasoning)} chars of reasoning:\n{reasoning[:500]}{'...' if len(reasoning) > 500 else ''}")

    return content
