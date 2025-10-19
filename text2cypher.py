#!/usr/bin/env python3
"""
text2cypher.py - Convert natural language queries to Cypher queries for educational knowledge graph
"""

import os
import json
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from neo4j import GraphDatabase
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks.manager import get_openai_callback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SchemaInfo:
    """Data class to hold Neo4j schema information"""
    node_labels: List[str]
    relationship_types: List[str]
    node_properties: Dict[str, List[str]]
    sample_nodes: Dict[str, List[Dict]]

class Neo4jSchemaExtractor:
    """Extract schema information from Neo4j database"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        """Close the Neo4j connection"""
        self.driver.close()
    
    def extract_schema(self) -> SchemaInfo:
        """Extract comprehensive schema information from the database"""
        with self.driver.session() as session:
            # Get node labels
            node_labels = session.run("CALL db.labels()").value()
            
            # Get relationship types
            relationship_types = session.run("CALL db.relationshipTypes()").value()
            
            # Get node properties for each label
            node_properties = {}
            sample_nodes = {}
            
            for label in node_labels:
                # Get properties for this label
                props_query = f"MATCH (n:{label}) RETURN DISTINCT keys(n) as props LIMIT 1"
                props_result = session.run(props_query).single()
                if props_result:
                    node_properties[label] = props_result["props"]
                else:
                    node_properties[label] = []
                
                # Get sample nodes for this label
                sample_query = f"MATCH (n:{label}) RETURN n LIMIT 3"
                samples = []
                for record in session.run(sample_query):
                    node_dict = dict(record["n"])
                    samples.append(node_dict)
                sample_nodes[label] = samples
        
        return SchemaInfo(
            node_labels=node_labels,
            relationship_types=relationship_types,
            node_properties=node_properties,
            sample_nodes=sample_nodes
        )

class Text2CypherConverter:
    """Main class for converting natural language to Cypher queries"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, openai_api_key: str):
        self.schema_extractor = Neo4jSchemaExtractor(neo4j_uri, neo4j_user, neo4j_password)
        self.schema_info = self.schema_extractor.extract_schema()
        
        # Initialize OpenAI LLM
        self.llm = OpenAI(
            openai_api_key=openai_api_key,
            temperature=0.1,  # Low temperature for more consistent outputs
            max_tokens=500
        )
        
        # Create the prompt template and chain using LCEL
        self.prompt_template = self._create_prompt_template()
        self.output_parser = StrOutputParser()
        self.chain = self.prompt_template | self.llm | self.output_parser
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create a comprehensive prompt template for text2cypher conversion"""
        
        # Build schema description
        schema_desc = self._build_schema_description()
        
        # Create few-shot examples based on the actual schema
        examples = self._create_few_shot_examples()
        
        # Build template without f-string to avoid variable interpolation issues
        template = """You are a Neo4j Cypher query expert for an educational knowledge graph system.

""" + schema_desc + """

IMPORTANT RULES:
1. Always use EXACT node labels and property names from the schema above
2. Use case-insensitive matching with toLower() for text searches
3. Return LIMIT 20 unless specified otherwise
4. For "what" questions, return node properties
5. For "how many" questions, use COUNT()
6. Use SUGGESTS relationship for positive recommendations
7. Use NO_SUGGESTS relationship for negative recommendations
8. Always return meaningful node properties like name and category

EXAMPLES:
""" + examples + """

QUERY PATTERNS:
- Student needs: MATCH (s:StudentWithSpecialNeeds)-[r:SUGGESTS]->(m:PedagogicalMethodology)
- Student characteristics: MATCH (s:StudentCharacteristic)-[r:SUGGESTS]->(p:PedagogicalApproach)
- Learning methods: MATCH (m:PedagogicalMethodology) WHERE toLower(m.name) CONTAINS toLower("keyword")
- Technology integration: MATCH (i:InteractiveBoard)-[r:SUPPORTS]->(p:PedagogicalApproach)
- Environmental factors: MATCH (e:LearningEnvironment)-[r:SUPPORTS]->(l:LearningProcess)
- Inclusion strategies: MATCH (p:PedagogicalStrategy)-[r:PROMOTES]->(i:InclusionStrategy)
- Learning barriers: MATCH (b:EnvironmentalBarrier)-[r:HINDERS]->(o:LearningOutcome)

Convert this natural language question to a Cypher query:
Question: {question}

Cypher Query:"""

        return PromptTemplate(
            input_variables=["question"],
            template=template
        )
    
    def _build_schema_description(self) -> str:
        """Build a comprehensive schema description"""
        desc = "DATABASE SCHEMA:\n"
        desc += "=" * 50 + "\n\n"
        
        desc += "NODE LABELS:\n"
        for label in self.schema_info.node_labels:
            desc += f"- {label}\n"
            if label in self.schema_info.node_properties:
                props = self.schema_info.node_properties[label]
                desc += f"  Properties: {', '.join(props)}\n"
            
            # Add sample data without dictionary representation
            if label in self.schema_info.sample_nodes and self.schema_info.sample_nodes[label]:
                sample = self.schema_info.sample_nodes[label][0]
                # Format sample data safely without single quotes
                sample_str = ", ".join([f"{k}: {v}" for k, v in sample.items()])
                desc += f"  Example: {sample_str}\n"
            desc += "\n"
        
        desc += "RELATIONSHIP TYPES:\n"
        for rel_type in self.schema_info.relationship_types:
            desc += f"- {rel_type}\n"
        
        return desc
    
    def _create_few_shot_examples(self) -> str:
        """Create few-shot examples based on the actual schema"""
        examples = """
Question: "What teaching methods help students with ADHD?"
Cypher: MATCH (s:StudentWithSpecialNeeds)-[r:SUGGESTS]->(m:PedagogicalMethodology) WHERE s.name = "Adhd" OR s.name = "Attention Deficit" RETURN m.name, m.category LIMIT 10

Question: "What pedagogical approaches should be avoided for students with no personal motivation?"
Cypher: MATCH (s:StudentWithSpecialNeeds)-[r:NO_SUGGESTS]->(m:PedagogicalMethodology) WHERE s.name = "NoPersonalMotivation" RETURN m.name, m.category LIMIT 10

Question: "How many pedagogical methodologies are in the database?"
Cypher: MATCH (m:PedagogicalMethodology) RETURN COUNT(m) as methodology_count

Question: "What lighting conditions support learning focus?"
Cypher: MATCH (l:Lighting)-[r:SUPPORTS]->(p:LearningProcess) WHERE p.name = "focus" RETURN l.name, p.name LIMIT 10

Question: "What colors facilitate attention and relaxation?"
Cypher: MATCH (c:Colour)-[r:FACILITATES]->(r:LearnerResponse) WHERE toLower(r.name) CONTAINS "attention" RETURN c.name, r.name LIMIT 10

Question: "What does interactive board enable for learning?"
Cypher: MATCH (i:InteractiveBoard)-[r:ENABLES]->(l:LearningModality) RETURN i.name, l.name LIMIT 10

Question: "What methodologies help students with excellence in subjects?"
Cypher: MATCH (s:StudentCharacteristic)-[r:SUGGESTS]->(m:PedagogicalMethodology) WHERE s.name = "Excellence in some or all subjects" RETURN m.name, m.category LIMIT 10

Question: "What furniture causes discomfort and reduced focus?"
Cypher: MATCH (f:Furniture)-[r:CAUSES]->(e:EnvironmentalBarrier) WHERE toLower(e.name) CONTAINS "discomfort" RETURN f.name, e.name LIMIT 10

Question: "What methodologies work for cohesive classes?"
Cypher: MATCH (c:Context)-[r:SUGGESTS]->(m:PedagogicalMethodology) WHERE c.name = "Cohesive" RETURN m.name, m.category LIMIT 10

Question: "What colors support cognitive regulation strategies?"
Cypher: MATCH (c:Colour)-[r:SUGGESTS]->(p:PedagogicalStrategy) WHERE toLower(p.name) CONTAINS "cognitive" RETURN c.name, p.name LIMIT 10

Question: "What environmental factors support learning environment?"
Cypher: MATCH (f:Furniture)-[r:SUPPORTS]->(e:LearningEnvironment) RETURN f.name, e.name LIMIT 10

Question: "What acoustic conditions increase cognitive load?"
Cypher: MATCH (a:Acoustic)-[r:INCREASES]->(c:CognitiveConstraint) RETURN a.name, c.name LIMIT 10
"""
        return examples.strip()
    
    def convert(self, question: str) -> Tuple[str, Dict]:
        """Convert natural language question to Cypher query"""
        try:
            with get_openai_callback() as cb:
                # Generate Cypher query using LCEL
                result = self.chain.invoke({"question": question})
                
                # Clean the result
                cypher_query = self._clean_cypher_query(result)
                
                # Repair common issues based on audit findings
                cypher_query = self._repair_cypher(cypher_query)
                
                # Validate the query
                is_valid, validation_error = self._validate_cypher(cypher_query)
                
                metadata = {
                    "tokens_used": cb.total_tokens,
                    "cost": cb.total_cost,
                    "is_valid": is_valid,
                    "validation_error": validation_error,
                    "original_question": question
                }
                
                return cypher_query, metadata
                
        except Exception as e:
            logger.error(f"Error converting question to Cypher: {e}")
            return "", {"error": str(e), "is_valid": False}
    
    def _clean_cypher_query(self, raw_query: str) -> str:
        """Clean and format the generated Cypher query"""
        # Remove any explanatory text before/after the query
        lines = raw_query.strip().split('\n')
        cypher_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and explanatory text
            if (line and 
                not line.startswith('Question:') and 
                not line.startswith('Answer:') and
                not line.startswith('Explanation:') and
                not line.startswith('Note:')):
                cypher_lines.append(line)
        
        query = ' '.join(cypher_lines)
        
        # Remove common prefixes
        prefixes_to_remove = [
            "Cypher Query:",
            "Cypher:",
            "Query:",
            "The Cypher query is:",
            "Here's the Cypher query:"
        ]
        
        for prefix in prefixes_to_remove:
            if query.startswith(prefix):
                query = query[len(prefix):].strip()
        
        return query
    
    def _repair_cypher(self, query: str) -> str:
        """Repair common Cypher query issues based on actual data patterns"""
        import re
        
        # 1) Force default pattern for StudentWithSpecialNeeds (validated pattern)
        # Change PedagogicalStrategy -> PedagogicalMethodology (our audit shows this is the correct pattern)
        query = re.sub(
            r'(\(s:StudentWithSpecialNeeds\)\s*-\s*\[r:\s*SUGGESTS\s*\]->\s*\()[a-zA-Z]*:PedagogicalStrategy(\))',
            r'\1m:PedagogicalMethodology\2',
            query
        )
        query = re.sub(
            r'(\(s:StudentWithSpecialNeeds\)\s*-\s*\[r:\s*NO_SUGGESTS\s*\]->\s*\()[a-zA-Z]*:PedagogicalStrategy(\))',
            r'\1m:PedagogicalMethodology\2',
            query
        )
        
        # 2) Remove subject-based constraints that don't exist in schema
        query = re.sub(r'AND\s+m\.name\s*=\s*"[Ss]cience[s]?"', '', query)
        query = re.sub(r'AND\s+m\.subject\s*=\s*"[^"]*"', '', query)
        query = re.sub(r'WHERE\s+m\.name\s*=\s*"[Ss]cience[s]?"\s+AND\s+', 'WHERE ', query)
        
        # 3) Fix bracketed cognitive disability cases: equality -> contains
        query = re.sub(
            r's\.name\s*=\s*"Cognitive disability \[mild, moderate, severe\]"',
            'toLower(s.name) CONTAINS "cognitive disability"',
            query,
            flags=re.IGNORECASE
        )
        
        # 4) Convert strict equality to case-insensitive for better matching
        # But preserve the synonym patterns we already have working
        pattern = r'WHERE\s+s\.name\s*=\s*"([^"]+)"(?!\s+OR\s+s\.name)'
        def make_case_insensitive(match):
            term = match.group(1)
            # Don't change if it's already part of a synonym list
            return f'WHERE toLower(s.name) = toLower("{term}")'
        query = re.sub(pattern, make_case_insensitive, query)
        
        # 5) Fix newlines in string literals
        query = query.replace("Assessment \nand evaluation", "Assessment and evaluation")
        query = query.replace("\n", " ")
        
        # 5.1) Fix missing closing parenthesis in WHERE clauses
        # Pattern: WHERE ... CONTAINS toLower("term" RETURN -> WHERE ... CONTAINS toLower("term") RETURN
        query = re.sub(r'CONTAINS toLower\("([^"]+)"\s+RETURN', r'CONTAINS toLower("\1") RETURN', query)
        
        # 5.2) Fix missing closing parenthesis in AND clauses
        # Pattern: AND toLower(s.name) CONTAINS toLower("term" RETURN -> AND toLower(s.name) CONTAINS toLower("term") RETURN
        query = re.sub(r'AND toLower\([^)]+\) CONTAINS toLower\("([^"]+)"\s+RETURN', r'AND toLower(s.name) CONTAINS toLower("\1") RETURN', query)
        
        # 5.3) Fix missing closing parenthesis in complex WHERE clauses
        # Pattern: CONTAINS toLower("term" AND -> CONTAINS toLower("term") AND
        query = re.sub(r'CONTAINS toLower\("([^"]+)"\s+AND\s+', r'CONTAINS toLower("\1") AND ', query)
        
        # 5.4) Fix missing closing parenthesis before RETURN
        # Pattern: CONTAINS toLower("term" RETURN -> CONTAINS toLower("term") RETURN
        query = re.sub(r'CONTAINS toLower\("([^"]+)"\s+RETURN', r'CONTAINS toLower("\1") RETURN', query)
        
        # 5.5) Fix the specific pattern from the error log
        # Pattern: toLower("Cognitive disability" RETURN -> toLower("Cognitive disability") RETURN
        query = re.sub(r'toLower\("([^"]+)"\s+RETURN', r'toLower("\1") RETURN', query)
        
        # 5.6) Fix nested CONTAINS without closing parenthesis before AND/RETURN
        # Pattern: CONTAINS toLower("term" AND -> CONTAINS toLower("term") AND
        query = re.sub(r'CONTAINS\s+toLower\("([^"]+)"\s+(AND|RETURN)', r'CONTAINS toLower("\1") \2', query)
        
        # 5.7) Fix complex WHERE with AND + CONTAINS missing closing parenthesis
        # Pattern: AND toLower(s.name) CONTAINS toLower("term" RETURN -> AND toLower(s.name) CONTAINS toLower("term") RETURN
        query = re.sub(
            r'AND\s+toLower\(([^)]+)\)\s+CONTAINS\s+toLower\("([^"]+)"\s+(AND|RETURN)',
            r'AND toLower(\1) CONTAINS toLower("\2") \3',
            query
        )
        
        # 6) Add synonym expansion for critical SEN terms
        query = self._expand_sen_synonyms(query)
        
        # 7) ChatGPT's specific fixes for remaining issues
        
        # A) Global Alias harmonization: fix any stray p.* to m.* when m is bound
        if re.search(r'\(\s*m\s*:\s*PedagogicalMethodology\s*\)', query) and not re.search(r'\(\s*p\s*:', query):
            # Replace ANY occurrence of 'p.' (not just right after RETURN) with 'm.'
            query = re.sub(r'(?<![A-Za-z0-9_])p\.', 'm.', query)

        # Optional: if LearningResource is the target, normalize p.* -> lr.*
        if re.search(r'\(\s*lr\s*:\s*LearningResource\s*\)', query) and not re.search(r'\(\s*p\s*:', query):
            query = re.sub(r'(?<![A-Za-z0-9_])p\.', 'lr.', query)
        
        # B) Cognitive OR-chain scrub: if we already have CONTAINS("cognitive disability"), drop any trailing OR s.name = "...Cognitive..."
        # First: Insert CONTAINS for cognitive disability patterns
        query = re.sub(
            r'WHERE\s+s\.name\s*=\s*"[^"]*[Cc]ognitive[^"]*"(?:\s+OR\s+s\.name\s*=\s*"[^"]*[Cc]ognitive[^"]*")*',
            'WHERE toLower(s.name) CONTAINS toLower("cognitive disability")',
            query
        )
        
        # Second: Clean up any remaining cognitive OR fragments after CONTAINS
        if re.search(r'toLower\(s\.name\)\s*CONTAINS\s*toLower\("cognitive disability"\)', query, flags=re.IGNORECASE):
            # Remove " OR s.name = "<anything with Cognitive ...>"
            query = re.sub(r'\s*OR\s*s\.name\s*=\s*"[^"]*(?i:cognitive)[^"]*"', '', query, flags=re.IGNORECASE)
            # Also remove accidental concatenation like ...CONTAINS("...") s.name = ...
            query = re.sub(r'\)\s*s\.name\s*=\s*"[^"]*(?i:cognitive)[^"]*"', ')', query, flags=re.IGNORECASE)
        
        # C) Fix Physical disability on wrong property
        query = re.sub(
            r's\.category\s*=\s*"(?i:physical disability)"',
            'toLower(s.name) CONTAINS toLower("Physical disability")',
            query,
            flags=re.IGNORECASE
        )
        
        # D) Relax LearningResource category when too strict
        # If we ask LR with a hard category filter, prefer a tolerant match
        query = re.sub(
            r'lr\.category\s*=\s*"(?i:assessment and evaluation)"',
            'toLower(lr.category) CONTAINS toLower("assessment")',
            query,
            flags=re.IGNORECASE
        )
        
        # E) Final hygiene: collapse duplicate spaces, fix WHERE ... AND/OR punctuation issues
        query = re.sub(r'\s+', ' ', query).strip()
        query = re.sub(r'WHERE\s+(AND|OR)\s+', 'WHERE ', query, flags=re.IGNORECASE)
        query = re.sub(r'\s+(AND|OR)\s+(RETURN|LIMIT)\b', r' \2', query, flags=re.IGNORECASE)
        
        return query
    
    def _expand_sen_synonyms(self, query: str) -> str:
        """Expand critical SEN terms with synonyms based on audit data"""
        import re
        
        # Synonym mappings based on actual node names in your data
        SEN_SYNONYMS = {
            "adhd": ["Adhd", "Attention Deficit", "Hyperactivity Disorder"],
            "attention deficit": ["Attention Deficit", "Adhd", "Hyperactivity Disorder"], 
            "autism spectrum disorder": ["Autism spectrum disorder"],
            "autism": ["Autism spectrum disorder"],
            "no personal motivation": ["NoPersonalMotivation", "Lack of motivation"],
            "nopersonalmotivation": ["NoPersonalMotivation", "Lack of motivation"],
            "cognitive disability": ["Cognitive disability [mild, moderate, severe]"],
            "physical disability": ["Physical disability"],
            "language difficulties": ["Language difficulties due to foreign origin"]
        }
        
        # Look for WHERE clauses with single term matching
        pattern = r'WHERE\s+toLower\(s\.name\)\s*=\s*toLower\("([^"]+)"\)'
        
        def expand_synonyms(match):
            term = match.group(1).lower()
            
            # Check if this term has synonyms
            for key, synonyms in SEN_SYNONYMS.items():
                if key in term or any(term in syn.lower() for syn in synonyms):
                    # Create IN clause with all synonyms
                    synonym_list = '", "'.join(synonyms)
                    return f'WHERE s.name IN ["{synonym_list}"]'
            
            # If no synonyms found, use tolerant CONTAINS matching
            return f'WHERE toLower(s.name) CONTAINS toLower("{match.group(1)}")'
        
        return re.sub(pattern, expand_synonyms, query)
    
    def _validate_cypher(self, cypher_query: str) -> Tuple[bool, Optional[str]]:
        """Validate the generated Cypher query syntax"""
        if not cypher_query or cypher_query.strip() == "":
            return False, "Empty query"
        
        try:
            # Basic syntax validation
            with self.schema_extractor.driver.session() as session:
                # Try to explain the query (doesn't execute it)
                session.run(f"EXPLAIN {cypher_query}")
                return True, None
                
        except Exception as e:
            return False, str(e)
    
    def execute_query(self, cypher_query: str) -> Tuple[List[Dict], Optional[str]]:
        """Execute the Cypher query and return results"""
        try:
            with self.schema_extractor.driver.session() as session:
                result = session.run(cypher_query)
                records = []
                for record in result:
                    records.append(dict(record))
                return records, None
                
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}")
            return [], str(e)
    
    def close(self):
        """Close all connections"""
        self.schema_extractor.close()

class Text2CypherPipeline:
    """Complete pipeline for text2cypher conversion and execution"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, openai_api_key: str):
        self.converter = Text2CypherConverter(neo4j_uri, neo4j_user, neo4j_password, openai_api_key)
    
    def process_question(self, question: str, execute: bool = True) -> Dict:
        """Process a natural language question end-to-end"""
        # Convert to Cypher
        cypher_query, metadata = self.converter.convert(question)
        
        result = {
            "question": question,
            "cypher_query": cypher_query,
            "metadata": metadata,
            "results": [],
            "execution_error": None
        }
        
        # Execute if requested and query is valid
        if execute and metadata.get("is_valid", False):
            results, execution_error = self.converter.execute_query(cypher_query)
            
            # Retry-on-empty orchestrator: if query is valid but returns 0 rows, try widened version
            if not results and not execution_error:
                # Create a more tolerant version of the query
                widened_query = cypher_query
                
                # Make name matching more tolerant
                widened_query = re.sub(r' s\.name IN \[', ' toLower(s.name) IN [', widened_query)
                widened_query = re.sub(r' s\.name = ', ' toLower(s.name) CONTAINS toLower(', widened_query)
                widened_query = re.sub(r' lr\.category = ', ' toLower(lr.category) CONTAINS toLower(', widened_query)
                
                # Only retry if the query actually changed
                if widened_query != cypher_query:
                    retry_results, retry_error = self.converter.execute_query(widened_query)
                    if retry_results:  # If retry succeeded, use its results
                        results = retry_results
                        execution_error = retry_error
                        result["retry_query"] = widened_query
                        result["retry_success"] = True
                    else:
                        result["retry_query"] = widened_query
                        result["retry_success"] = False
            
            result["results"] = results
            result["execution_error"] = execution_error
        
        return result
    
    def close(self):
        """Close all connections"""
        self.converter.close()

# Example usage and testing functions
def main():
    """Example usage of the Text2Cypher module"""
    # Configuration
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "your_password"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    # Initialize pipeline
    pipeline = Text2CypherPipeline(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY)
    
    # Test questions
    test_questions = [
        "What teaching methods help students with ADHD?",
        "What methodologies should be avoided for students with autism?",
        "How many different teaching methodologies are available?",
        "What technologies support interactive learning?",
        "Show me learning environment factors for better acoustics"
    ]
    
    try:
        for question in test_questions:
            print(f"\n{'='*60}")
            print(f"Question: {question}")
            print('='*60)
            
            result = pipeline.process_question(question)
            
            print(f"Generated Cypher: {result['cypher_query']}")
            print(f"Valid: {result['metadata'].get('is_valid', False)}")
            
            if result['execution_error']:
                print(f"Execution Error: {result['execution_error']}")
            else:
                print(f"Results ({len(result['results'])} records):")
                for i, record in enumerate(result['results'][:3]):  # Show first 3
                    print(f"  {i+1}. {record}")
                if len(result['results']) > 3:
                    print(f"  ... and {len(result['results']) - 3} more")
    
    finally:
        pipeline.close()

if __name__ == "__main__":
    main() 