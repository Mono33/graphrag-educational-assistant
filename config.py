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

@dataclass
class OpenAIConfig:
    """OpenAI API configuration"""
    api_key: str = ""
    model: str = "gpt-3.5-turbo-instruct"  # For LangChain OpenAI LLM
    temperature: float = 0.1
    max_tokens: int = 500

@dataclass
class Text2CypherConfig:
    """Text2Cypher module configuration"""
    max_query_length: int = 1000
    default_limit: int = 20
    enable_query_validation: bool = True
    enable_query_execution: bool = True
    log_level: str = "INFO"

class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.neo4j = Neo4jConfig()
        self.openai = OpenAIConfig()
        self.text2cypher = Text2CypherConfig()
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # Neo4j configuration
        self.neo4j.uri = os.getenv("NEO4J_URI", self.neo4j.uri)
        self.neo4j.user = os.getenv("NEO4J_USER", self.neo4j.user)
        self.neo4j.password = os.getenv("NEO4J_PASSWORD", self.neo4j.password)
        self.neo4j.database = os.getenv("NEO4J_DATABASE", self.neo4j.database)
        
        # OpenAI configuration
        self.openai.api_key = os.getenv("OPENAI_API_KEY", self.openai.api_key)
        self.openai.model = os.getenv("OPENAI_MODEL", self.openai.model)
        
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