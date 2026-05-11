"""
Neo4j tool — executes arbitrary Cypher queries and returns rows as JSON-serializable dicts.
Used by the LM Studio agent to explore the Knowledge Graph.
"""

import os
import logging
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_driver = None


def _get_driver():
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase

        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        if not uri or not password:
            raise RuntimeError("NEO4J_URI and NEO4J_PASSWORD must be set in .env")
        _driver = GraphDatabase.driver(uri, auth=(user, password))
    return _driver


def run_query(cypher: str, limit_rows: int = 50) -> Dict[str, Any]:
    """
    Execute a Cypher query and return up to limit_rows rows.

    Returns:
        {"rows": [...], "count": int} on success
        {"error": str} on failure
    """
    try:
        driver = _get_driver()
        database = os.getenv("NEO4J_DATABASE", "neo4j")
        with driver.session(database=database) as session:
            result = session.run(cypher)
            rows = []
            for record in result:
                row = {}
                for key in record.keys():
                    value = record[key]
                    # Convert Neo4j types to plain Python
                    if hasattr(value, "__class__") and value.__class__.__name__ in ("Node", "Relationship"):
                        row[key] = dict(value)
                    elif isinstance(value, (list, tuple)):
                        row[key] = list(value)
                    else:
                        row[key] = value
                rows.append(row)
                if len(rows) >= limit_rows:
                    break
        return {"rows": rows, "count": len(rows)}
    except Exception as e:
        logger.warning(f"[neo4j_tool] Query failed: {e}")
        return {"error": str(e)}


def close():
    global _driver
    if _driver:
        _driver.close()
        _driver = None


# Tool definition for OpenAI function-calling format
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "query_neo4j",
        "description": (
            "Run a read-only Cypher query against the Neo4j Knowledge Graph. "
            "Use this to understand a concept's relationships, labels, domain, and educational context. "
            "Example: MATCH (n {name: 'Metacognition'})-[r]-(m) RETURN type(r) AS rel, m.name AS neighbor, labels(m) AS labels LIMIT 20"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cypher": {
                    "type": "string",
                    "description": (
                        "Cypher query to execute. Always include LIMIT to avoid large result sets. "
                        "Useful patterns:\n"
                        "  - Concept neighborhood: MATCH (n {name: $name})-[r]-(m) RETURN type(r), m.name, labels(m) LIMIT 20\n"
                        "  - All concepts in domain: MATCH (n:NeuroNode) RETURN n.name LIMIT 50\n"
                        "  - Specific relationship: MATCH (a)-[:SUGGESTS]->(b) RETURN a.name, b.name LIMIT 20"
                    ),
                }
            },
            "required": ["cypher"],
        },
    },
}
