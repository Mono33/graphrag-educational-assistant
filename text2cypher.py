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
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
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
    
    def extract_schema(self, domain: str = None) -> SchemaInfo:
        """Extract comprehensive schema information from the database
        
        Args:
            domain: Optional domain filter (e.g., 'udl', 'neuro'). 
                   If None, extracts all labels.
        """
        with self.driver.session() as session:
            # Get node labels (domain-filtered if specified)
            if domain and domain != "all":
                # Extract ONLY labels for this domain
                node_labels = session.run("""
                    MATCH (n {domain: $domain})
                    RETURN DISTINCT labels(n)[0] as label
                    ORDER BY label
                """, domain=domain).value()
            else:
                # Get all labels (for cross-domain or backward compatibility)
                node_labels = session.run("CALL db.labels()").value()
            
            # Get relationship types (domain-filtered if specified)
            if domain and domain != "all":
                relationship_types = session.run("""
                    MATCH (a {domain: $domain})-[r]->(b {domain: $domain})
                    RETURN DISTINCT type(r) as rel_type
                    ORDER BY rel_type
                """, domain=domain).value()
            else:
                relationship_types = session.run("CALL db.relationshipTypes()").value()
            
            # Get node properties for each label
            node_properties = {}
            sample_nodes = {}
            
            for label in node_labels:
                # Get properties for this label (domain-filtered)
                if domain and domain != "all":
                    props_query = f"MATCH (n:{label} {{domain: $domain}}) RETURN DISTINCT keys(n) as props LIMIT 1"
                    props_result = session.run(props_query, domain=domain).single()
                else:
                    props_query = f"MATCH (n:{label}) RETURN DISTINCT keys(n) as props LIMIT 1"
                    props_result = session.run(props_query).single()
                    
                if props_result:
                    node_properties[label] = props_result["props"]
                else:
                    node_properties[label] = []
                
                # Get sample nodes for this label (domain-filtered)
                if domain and domain != "all":
                    sample_query = f"MATCH (n:{label} {{domain: $domain}}) RETURN n LIMIT 3"
                    sample_records = session.run(sample_query, domain=domain)
                else:
                    sample_query = f"MATCH (n:{label}) RETURN n LIMIT 3"
                    sample_records = session.run(sample_query)
                    
                samples = []
                for record in sample_records:
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
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, openai_api_key: str, model: str = "gpt-3.5-turbo-16k"):
        self.schema_extractor = Neo4jSchemaExtractor(neo4j_uri, neo4j_user, neo4j_password)
        # Don't extract schema in __init__ anymore - do it per-query with domain filter
        self.schema_info = None  # Will be set per-query
        
        # Initialize OpenAI LLM (using Chat API for larger token limits)
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model=model,  # Uses model from config (e.g., gpt-4o from .env)
            temperature=0.1,  # Low temperature for deterministic Cypher generation
            max_tokens=500  # Sufficient for Cypher queries
        )
        
        self.output_parser = StrOutputParser()
        # Prompt template and chain will be created per-query with domain-specific schema
    
    def _create_prompt_template(self, domain: str = None) -> ChatPromptTemplate:
        """Create a comprehensive prompt template for text2cypher conversion
        
        Args:
            domain: Domain for which to create the prompt (e.g., 'udl', 'neuro')
        """
        
        # Build schema description (domain-specific)
        schema_desc = self._build_schema_description()
        
        # Create domain-specific few-shot examples
        if domain == "udl":
            examples = self._get_udl_examples()
        elif domain == "neuro":
            examples = self._get_neuro_examples(domain)
        else:
            examples = self._create_few_shot_examples()  # Generic fallback
        
        # Build system message with schema and instructions
        system_message = """You are a Neo4j Cypher query expert for an educational knowledge graph system.

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
9. CRITICAL: For queries about teaching methods/strategies/approaches for students with special needs, ALWAYS use pattern:
   (s:StudentWithSpecialNeeds)-[r:SUGGESTS]->(m:PedagogicalMethodology)
   NOT InclusionStrategy or other node types!
10. CRITICAL OUTPUT FORMAT: Return ONLY valid Cypher code, NO explanations, NO apologies, NO "I'm sorry" messages!
    - ❌ BAD: "I'm sorry, but... Cypher Query: MATCH..."
    - ✅ GOOD: "MATCH (n) RETURN n"
"""
        
        # Add domain-specific rules
        if domain == "neuro":
            system_message += """
CRITICAL NEUROSCIENCE DOMAIN RULES (MUST FOLLOW):
- You are querying NEUROSCIENCE CONCEPTS that inform educational practices
- This knowledge graph contains research concepts AND teaching practices grounded in neuroscience
- Available teaching-related labels: TeachingPractices (use this for classroom strategies!)
- NEVER use these UDL-only labels: PedagogicalMethodology, StudentWithSpecialNeeds, StudentCharacteristic, Context, Lighting, Furniture, Assessment
- NEVER invent labels like: ClassroomActivity, Homework, Lesson, Exercise
- For "how-to" or "quali attività" questions (classroom strategies):
  * FIRST try: MATCH (t:TeachingPractices)-[r]-(stress_concept) for teaching practices
  * SECOND try: MATCH concepts that inform teaching (e.g., PositiveStressEustress, Metacognition)
  * ALWAYS generate valid Cypher - map practical questions to neuroscience concepts
  * Example: "reduce stress" → Query PositiveStressEustress, NegativeStressDistress, TeachingPractices
  * Example: "improve attention" → Query Attention, ExecutiveFunctions, TeachingPractices
- ONLY use neuroscience labels from the schema above: IntrinsicMotivation, GrowthMindset, Attention, ExecutiveFunctions, etc.
- CRITICAL: The schema above contains ALL valid labels for this domain. Do NOT invent or guess label names.
- If you cannot find a suitable neuroscience label for the concept, return a simple definition query for related concepts instead of inventing labels or relationships
- LABEL MAPPING EXAMPLES (use these when you cannot find exact match):
  * "goal setting" / "definizione obiettivi" → Use Planning or Metacognition (NOT GoalSetting or GoalSettingAndStrategySelectionBeforeATask)
  * "teaching methods" / "insegnamento" → Use TeachingPractices (NOT TeachingMethods or TeachingStrategies)
  * "emotions" / "emozioni" → Use PositiveEmotions or NegativeEmotions (NOT generic Emotions)
  * "memory" / "memoria" → Use Memory, WorkingMemory, or LongTermMemory (choose based on context)
- If you create a label that's longer than 3 words OR contains "And", you probably invented it - STOP and use a simpler label from the schema
- Focus on "what is X" and "how does X affect Y" patterns - these are well-supported

EXAMPLES:
""" + examples + """
"""
        else:
            system_message += """

EXAMPLES:
""" + examples + """
"""
        
        # Add domain-specific query patterns
        if domain == "udl":
            system_message += """
QUERY PATTERNS (UDL):
- Student needs: MATCH (s:StudentWithSpecialNeeds)-[r:SUGGESTS]->(m:PedagogicalMethodology)
- Student characteristics: MATCH (s:StudentCharacteristic)-[r:SUGGESTS]->(p:PedagogicalApproach)
- Learning methods: MATCH (m:PedagogicalMethodology) WHERE toLower(m.name) CONTAINS toLower("keyword")
- Technology integration: MATCH (i:InteractiveBoard)-[r:SUPPORTS]->(p:PedagogicalApproach)
- Environmental factors: MATCH (e:LearningEnvironment)-[r:SUPPORTS]->(l:LearningProcess)
- Inclusion strategies: MATCH (p:PedagogicalStrategy)-[r:PROMOTES]->(i:InclusionStrategy)
- Learning barriers: MATCH (b:EnvironmentalBarrier)-[r:HINDERS]->(o:LearningOutcome)
"""
        elif domain == "neuro":
            system_message += """
QUERY PATTERNS (NEURO):
- Concept definitions: MATCH (c:ConceptLabel) RETURN c.name, c.category
- Concept comparisons: MATCH (c1:Label1) RETURN ... UNION MATCH (c2:Label2) RETURN ...
- Concept relationships: MATCH (a:Label1)-[r:RELATIONSHIP]->(b:Label2) RETURN a.name, type(r), b.name
- Do NOT create patterns with teaching/strategy/activity labels
"""
        else:
            system_message += """
QUERY PATTERNS:
- Concept definitions: MATCH (c:Label) RETURN c.name, c.category
- Relationships: MATCH (a:Label1)-[r]->(b:Label2) RETURN a.name, type(r), b.name
"""
        
        system_message += """
Convert the natural language question to a Cypher query. Return ONLY the Cypher query, no explanations."""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "Question: {question}\n\nCypher Query:")
        ])
    
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
    
    def _get_udl_examples(self) -> str:
        """Get UDL-specific few-shot examples"""
        examples = """
Question: "What teaching methods help students with ADHD?"
Cypher: MATCH (s:StudentWithSpecialNeeds)-[r:SUGGESTS]->(m:PedagogicalMethodology) WHERE s.name = "Adhd" OR s.name = "Attention Deficit" RETURN m.name, m.category LIMIT 10

Question: "What strategies help blind students?"
Cypher: MATCH (s:StudentWithSpecialNeeds)-[r:SUGGESTS]->(m:PedagogicalMethodology) WHERE s.name = "Blind" RETURN m.name, m.category LIMIT 10

Question: "What approaches work for deaf students?"
Cypher: MATCH (s:StudentWithSpecialNeeds)-[r:SUGGESTS]->(m:PedagogicalMethodology) WHERE s.name = "Deaf" RETURN m.name, m.category LIMIT 10

Question: "What methods help students with autism spectrum disorder?"
Cypher: MATCH (s:StudentWithSpecialNeeds)-[r:SUGGESTS]->(m:PedagogicalMethodology) WHERE s.name = "Autism spectrum disorder" RETURN m.name, m.category LIMIT 10

Question: "What pedagogical approaches should be avoided for students with no personal motivation?"
Cypher: MATCH (s:StudentWithSpecialNeeds)-[r:NO_SUGGESTS]->(m:PedagogicalMethodology) WHERE s.name = "NoPersonalMotivation" RETURN m.name, m.category LIMIT 10

Question: "How many pedagogical methodologies are in the database?"
Cypher: MATCH (m:PedagogicalMethodology) RETURN COUNT(m) as methodology_count

Question: "What lighting conditions support learning focus?"
Cypher: MATCH (l:Lighting)-[r:SUPPORTS]->(p:LearningProcess) WHERE p.name = "focus" RETURN l.name, p.name LIMIT 10

Question: "What colors facilitate attention and relaxation?"
Cypher: MATCH (c:Colour)-[r:FACILITATES]->(r:LearnerResponse) WHERE toLower(r.name) CONTAINS "attention" RETURN c.name, r.name LIMIT 10

Question: "What methodologies help students with excellence in subjects?"
Cypher: MATCH (s:StudentCharacteristic)-[r:SUGGESTS]->(m:PedagogicalMethodology) WHERE s.name = "Excellence in some or all subjects" RETURN m.name, m.category LIMIT 10

Question: "What furniture causes discomfort and reduced focus?"
Cypher: MATCH (f:Furniture)-[r:CAUSES]->(e:EnvironmentalBarrier) WHERE toLower(e.name) CONTAINS "discomfort" RETURN f.name, e.name LIMIT 10

Question: "What methodologies work for cohesive classes?"
Cypher: MATCH (c:Context)-[r:SUGGESTS]->(m:PedagogicalMethodology) WHERE c.name = "Cohesive" RETURN m.name, m.category LIMIT 10
"""
        return examples.strip()
    
    def _get_neuro_examples(self, domain: str = "neuro") -> str:
        """Get Neuro-specific few-shot examples based on actual Neuro data (Nov 2025)
        
        All patterns verified against neuro_audit_report.json:
        - 478 nodes, 195 unique labels, 111 relationship types
        - Top relationships: SUPPORTS (41), ENHANCES (37), ENHANCE (29), LEADS_TO (22)
        - Focus on definition, comparison, and relationship queries
        """
        # Use regular string and replace {DOMAIN} placeholder
        examples = """
Question: "What is intrinsic motivation?"
Cypher: MATCH (m:IntrinsicMotivation {{domain: "{domain}"}}) RETURN m, labels(m) as node_labels LIMIT 10

Question: "What is extrinsic motivation?"
Cypher: MATCH (m:ExtrinsicMotivation {{domain: "{domain}"}}) RETURN m, labels(m) as node_labels LIMIT 10

Question: "What is the difference between intrinsic and extrinsic motivation?"
Cypher: MATCH (m:IntrinsicMotivation {{domain: "{domain}"}}) RETURN "Intrinsic" as type, m, labels(m) as node_labels UNION MATCH (m:ExtrinsicMotivation {{domain: "{domain}"}}) RETURN "Extrinsic" as type, m, labels(m) as node_labels LIMIT 20

Question: "What is growth mindset?"
Cypher: MATCH (g:GrowthMindset {{domain: "{domain}"}}) RETURN g, labels(g) as node_labels LIMIT 10

Question: "What is fixed mindset?"
Cypher: MATCH (f:FixedMindset {{domain: "{domain}"}}) RETURN f, labels(f) as node_labels LIMIT 10

Question: "What is the difference between growth mindset and fixed mindset?"
Cypher: MATCH (g:GrowthMindset {{domain: "{domain}"}}) RETURN "Growth" as type, g, labels(g) as node_labels UNION MATCH (f:FixedMindset {{domain: "{domain}"}}) RETURN "Fixed" as type, f, labels(f) as node_labels LIMIT 20

Question: "What is positive stress?"
Cypher: MATCH (s:PositiveStressEustress {{domain: "{domain}"}}) RETURN s, labels(s) as node_labels LIMIT 10

Question: "What is the difference between positive stress and negative stress?"
Cypher: MATCH (s:PositiveStressEustress {{domain: "{domain}"}}) RETURN "Positive Stress" as type, s, labels(s) as node_labels UNION MATCH (s:NegativeStressDistress {{domain: "{domain}"}}) RETURN "Negative Stress" as type, s, labels(s) as node_labels LIMIT 20

Question: "What is attention?"
Cypher: MATCH (a:Attention {{domain: "{domain}"}}) RETURN a, labels(a) as node_labels LIMIT 10

Question: "What is selective attention?"
Cypher: MATCH (a:Attention {{domain: "{domain}", name: "Selective Attention"}}) RETURN a, labels(a) as node_labels

Question: "What are executive functions?"
Cypher: MATCH (e:ExecutiveFunctions {{domain: "{domain}"}}) RETURN e, labels(e) as node_labels LIMIT 10

Question: "What is working memory?"
Cypher: MATCH (w:WorkingMemory {{domain: "{domain}"}}) RETURN w, labels(w) as node_labels LIMIT 10

Question: "What is critical thinking?"
Cypher: MATCH (c:CriticalThinking {{domain: "{domain}"}}) RETURN c, labels(c) as node_labels LIMIT 10

Question: "What is metacognition?"
Cypher: MATCH (m:Metacognition {{domain: "{domain}"}}) RETURN m, labels(m) as node_labels LIMIT 10

Question: "What is emotional regulation?"
Cypher: MATCH (e:EmotionalRegulation {{domain: "{domain}"}}) RETURN e, labels(e) as node_labels LIMIT 10

Question: "How does attention support learning?"
Cypher: MATCH (a:Attention {{domain: "{domain}"}})-[r:SUPPORTS]->(o:OptimalAttentionalNetworkActivation {{domain: "{domain}"}}) RETURN a, type(r), o, labels(a) as source_labels, labels(o) as target_labels LIMIT 10

Question: "How does intrinsic motivation enhance executive functions?"
Cypher: MATCH (i:IntrinsicMotivation {{domain: "{domain}"}})-[r:ENHANCES]->(e:ExecutiveFunctions {{domain: "{domain}"}}) RETURN i, type(r), e, labels(i) as source_labels, labels(e) as target_labels LIMIT 10

Question: "What does extrinsic motivation reduce?"
Cypher: MATCH (e:ExtrinsicMotivation {{domain: "{domain}"}})-[r:REDUCES]->(target {{domain: "{domain}"}}) RETURN e, type(r), target, labels(e) as source_labels, labels(target) as target_labels LIMIT 10

Question: "What leads to learning development?"
Cypher: MATCH (source {{domain: "{domain}"}})-[r:LEADS_TO]->(l:LearningDevelopment {{domain: "{domain}"}}) RETURN source, type(r), l, labels(source) as source_labels, labels(l) as target_labels LIMIT 10

Question: "How does negative stress affect learning?"
Cypher: MATCH (n:NegativeStressDistress {{domain: "{domain}"}})-[r:UNDERMINES|LEADS_TO]->(target {{domain: "{domain}"}}) RETURN n, type(r), target, labels(n) as source_labels, labels(target) as target_labels LIMIT 10

Question: "What supports engagement?"
Cypher: MATCH (source {{domain: "{domain}"}})-[r:SUPPORTS|DRIVES]->(k:KnowledgeConstructionAttention {{domain: "{domain}"}}) RETURN source, type(r), k, labels(source) as source_labels, labels(k) as target_labels LIMIT 10

Question: "How does creativity enhance learning?"
Cypher: MATCH (c:Creativity {{domain: "{domain}"}})-[r:ENHANCE|FACILITATES]->(target {{domain: "{domain}"}}) RETURN c, type(r), target, labels(c) as source_labels, labels(target) as target_labels LIMIT 10

Question: "What are the positive emotions?"
Cypher: MATCH (p:PositiveEmotions {{domain: "{domain}"}}) RETURN p, labels(p) as node_labels LIMIT 10

Question: "How do positive emotions enhance cognition?"
Cypher: MATCH (p:PositiveEmotions {{domain: "{domain}"}})-[r:ENHANCE|ENHANCES]->(c:CognitiveProcesses {{domain: "{domain}"}}) RETURN p, type(r), c, labels(p) as source_labels, labels(c) as target_labels LIMIT 10

Question: "What is the relationship between mindset and learning?"
Cypher: MATCH (m:Mindset {{domain: "{domain}"}})-[r]->(target {{domain: "{domain}"}}) RETURN m, type(r), target, labels(m) as source_labels, labels(target) as target_labels LIMIT 10

Question: "Come posso migliorare l'attenzione degli studenti?"
Cypher: MATCH (a:Attention {{domain: "{domain}"}}) RETURN a, labels(a) as node_labels LIMIT 10

Question: "Quali sono i fattori che influenzano la memoria di lavoro?"
Cypher: MATCH (w:WorkingMemory {{domain: "{domain}"}}) RETURN w, labels(w) as node_labels LIMIT 10

Question: "Come migliorare la motivazione intrinseca?"
Cypher: MATCH (i:IntrinsicMotivation {{domain: "{domain}"}}) RETURN i, labels(i) as node_labels LIMIT 10
"""
        # Replace {domain} placeholder with actual domain value
        return examples.replace("{domain}", domain).strip()
    
    def _create_few_shot_examples(self) -> str:
        """Create generic few-shot examples (fallback)"""
        # Combine both UDL and Neuro examples for cross-domain
        udl_examples = self._get_udl_examples()
        neuro_examples = self._get_neuro_examples()
        return udl_examples + "\n\n" + neuro_examples
    
    def convert(self, question: str, domain: str = None) -> Tuple[str, Dict]:
        """Convert natural language question to Cypher query
        
        Args:
            question: Natural language question
            domain: Domain filter ('udl', 'neuro', 'all', or None)
        """
        try:
            # Extract domain-specific schema
            self.schema_info = self.schema_extractor.extract_schema(domain=domain)
            
            # Create domain-specific prompt template
            prompt_template = self._create_prompt_template(domain=domain)
            
            # Create chain for this query
            chain = prompt_template | self.llm | self.output_parser
            
            with get_openai_callback() as cb:
                # Generate Cypher query using LCEL
                result = chain.invoke({"question": question, "domain": domain})
                
                # Clean the result
                cypher_query = self._clean_cypher_query(result)
                
                # Detect UDL contamination (logging only, non-breaking)
                has_contamination = self._detect_udl_contamination(cypher_query, domain)
                
                # Apply domain-specific repairs
                cypher_query = self._repair_cypher(cypher_query, domain=domain)
                
                # Validate labels against schema (detect invented labels)
                self._validate_labels_against_schema(cypher_query, domain)
                
                # Inject domain filter (unless cross-domain)
                if domain and domain != "all":
                    cypher_query = self._inject_domain_filter(cypher_query, domain)
                
                # Log the generated query for debugging
                logger.info(f"Generated Cypher (domain={domain}): {cypher_query}")
                
                # Validate the query
                is_valid, validation_error = self._validate_cypher(cypher_query)
                
                metadata = {
                    "tokens_used": cb.total_tokens,
                    "cost": cb.total_cost,
                    "is_valid": is_valid,
                    "validation_error": validation_error,
                    "original_question": question,
                    "domain": domain,
                    "has_udl_contamination": has_contamination
                }
                
                return cypher_query, metadata
                
        except Exception as e:
            logger.error(f"Error converting question to Cypher: {e}")
            return "", {"error": str(e), "is_valid": False, "domain": domain}
    
    def _clean_cypher_query(self, raw_query: str) -> str:
        """Clean and format the generated Cypher query"""
        # Remove markdown code fences if present
        query = raw_query.strip()
        
        # Remove opening code fence (```cypher or ```)
        if query.startswith('```cypher'):
            query = query[9:].strip()  # Remove ```cypher
        elif query.startswith('```'):
            query = query[3:].strip()  # Remove ```
        
        # Remove closing code fence (```)
        if query.endswith('```'):
            query = query[:-3].strip()  # Remove trailing ```
        
        # Remove any explanatory text before/after the query
        lines = query.split('\n')
        cypher_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and explanatory text
            if (line and 
                not line.startswith('Question:') and 
                not line.startswith('Answer:') and
                not line.startswith('Explanation:') and
                not line.startswith('Note:') and
                not line.startswith('```')):  # Skip any remaining code fences
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
    
    def _inject_domain_filter(self, query: str, domain: str) -> str:
        """Inject domain filter into all MATCH clauses
        
        Args:
            query: Cypher query
            domain: Domain to filter by (e.g., 'udl', 'neuro')
        """
        import re
        
        # Pattern 1: MATCH (var:Label) -> MATCH (var:Label {domain: "domain"})
        # This handles simple single-node patterns
        pattern1 = r'\((\w+):(\w+)\)(?!\s*\{)'
        replacement1 = r'(\1:\2 {domain: "' + domain + '"})'
        query = re.sub(pattern1, replacement1, query)
        
        # Pattern 2: MATCH (var:Label {existing_prop: value}) -> add domain to existing props
        # This handles nodes that already have property filters
        pattern2 = r'\((\w+):(\w+)\s*\{([^}]+)\}\)'
        def add_domain_to_props(match):
            var = match.group(1)
            label = match.group(2)
            props = match.group(3)
            # Only add domain if not already present
            if 'domain:' not in props and 'domain =' not in props:
                return f'({var}:{label} {{{props}, domain: "{domain}"}})'
            return match.group(0)
        query = re.sub(pattern2, add_domain_to_props, query)
        
        return query
    
    def _repair_cypher(self, query: str, domain: str = None) -> str:
        """Repair common Cypher query issues based on actual data patterns
        
        Args:
            query: Cypher query to repair
            domain: Domain being queried (for domain-specific repairs)
        """
        import re
        
        # Route to domain-specific repairs
        if domain == "udl":
            query = self._repair_udl_query(query)
        elif domain == "neuro":
            query = self._repair_neuro_query(query)
        
        # Apply common repairs (all domains)
        query = self._repair_common_issues(query)
        
        return query
    
    def _repair_udl_query(self, query: str) -> str:
        """Apply UDL-specific query repairs and optimizations"""
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
        # PedagogicalMethodology nodes don't have subject properties - they're teaching methods, not subjects
        query = re.sub(r'AND\s+m\.name\s*=\s*"[Ss]cience[s]?"', '', query)
        query = re.sub(r'AND\s+m\.name\s*=\s*"[Mm]athematics?"', '', query)  # ✅ Added mathematics
        query = re.sub(r'AND\s+m\.name\s*=\s*"[Hh]istory"', '', query)  # ✅ Added history
        query = re.sub(r'AND\s+m\.name\s*=\s*"[Gg]eography"', '', query)  # ✅ Added geography
        query = re.sub(r'AND\s+m\.name\s*=\s*"[Ee]nglish"', '', query)  # ✅ Added English
        query = re.sub(r'AND\s+m\.subject\s*=\s*"[^"]*"', '', query)
        query = re.sub(r'WHERE\s+m\.name\s*=\s*"[Ss]cience[s]?"\s+AND\s+', 'WHERE ', query)
        query = re.sub(r'WHERE\s+m\.name\s*=\s*"[Mm]athematics?"\s+AND\s+', 'WHERE ', query)  # ✅ Added
        
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
        
        # 5.0) AGGRESSIVE: Fix ALL missing closing parentheses in toLower() before any keyword
        # This catches cases like: toLower("term" RETURN or toLower("term" AND or toLower("term" LIMIT
        query = re.sub(r'toLower\("([^"]+)"\s+(RETURN|AND|OR|LIMIT|WHERE)\b', r'toLower("\1") \2', query)
        
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
            "blind": ["Blind"],  # ✅ Added for ipovedenti queries
            "visually impaired": ["Blind"],  # ✅ Alternative term
            "deaf": ["Deaf"],  # ✅ Added for hearing impaired
            "hearing impaired": ["Deaf"],  # ✅ Alternative term
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
    
    def _detect_udl_contamination(self, query: str, domain: str) -> bool:
        """Detect UDL-specific labels in Neuro queries (non-breaking, logging only)
        
        Args:
            query: Cypher query to check
            domain: Domain being queried
            
        Returns:
            True if UDL contamination detected, False otherwise
        """
        if domain != "neuro":
            return False  # Only check contamination in Neuro domain
        
        # UDL-only labels (verified from audit reports)
        UDL_ONLY_LABELS = [
            'PedagogicalMethodology',  # Core UDL teaching methods
            'StudentWithSpecialNeeds',  # UDL special education
            'Lighting', 'Colour', 'Furniture', 'Texture', 'Smell',  # Environmental factors (UDL)
            'Acoustic', 'LearningEnvironment', 'EnvironmentalBarrier',  # Environmental (UDL)
            'Context', 'StudentCharacteristic',  # UDL-specific context
            'LearningProcess', 'LearnerResponse',  # UDL learning patterns
            'InteractiveBoard', 'Infrastructure',  # UDL technology
            'InclusionStrategy', 'PedagogicalStrategy',  # UDL strategies
        ]
        
        contamination_found = False
        for label in UDL_ONLY_LABELS:
            if f':{label}' in query:
                logger.warning(f"[UDL Contamination] Label '{label}' detected in Neuro query - this may return 0 results")
                contamination_found = True
        
        return contamination_found
    
    def _validate_labels_against_schema(self, query: str, domain: str) -> None:
        """Validate that all labels in query exist in domain schema (logging only)
        
        Args:
            query: Cypher query to validate
            domain: Domain being queried
        """
        if not self.schema_info:
            return
        
        import re
        
        # Extract ONLY node labels from query pattern (var:Label), not relationship types
        # Pattern: (var:Label) but NOT [r:RELATIONSHIP]
        node_label_pattern = r'\((?:\w+):(\w+)(?:\s*\{[^}]*\})?\)'
        labels_in_query = set(re.findall(node_label_pattern, query))
        
        # Get valid labels for this domain
        valid_labels = set(self.schema_info.node_labels)
        
        # Check for invalid labels
        for label in labels_in_query:
            if label not in valid_labels:
                logger.warning(
                    f"[Invalid Label] '{label}' does not exist in {domain} domain schema. "
                    f"Query will return 0 results. This may indicate a data gap or invented label."
                )
    
    def _repair_neuro_query(self, query: str) -> str:
        """Apply Neuro-specific query repairs and optimizations
        
        Based on neuro_audit_report.json (Nov 2025):
        - 478 nodes, 195 unique labels, 111 relationship types
        - Most common relationships: SUPPORTS, ENHANCES, ENHANCE, LEADS_TO, IS_LINKED_WITH
        """
        import re
        
        # 0) Fix variable name conflicts in relationships (CRITICAL FIX - Query 4 issue)
        # Pattern: MATCH (i:Label1)-[r:REL]->(r:Label2) - variable 'r' used for both relationship and node
        conflict_pattern = r'\[(\w+):([^\]]+)\]->\((\1):(\w+)'
        if re.search(conflict_pattern, query):
            old_var = re.search(conflict_pattern, query).group(1)
            logger.info(f"Fixing variable name conflict in relationship query (old var: '{old_var}')")
            
            # Step 1: Replace the conflicting pattern in MATCH clause
            # [r:REL]->(r:Label) → [rel:REL]->(target:Label)
            query = re.sub(
                r'\[(\w+):([^\]]+)\]->\((\1):(\w+)([^)]*)\)',
                r'[rel:\2]->(target:\4\5)',
                query
            )
            
            # Step 2: Fix RETURN clause - use simple, explicit replacements
            # Strategy: Replace the ENTIRE common patterns, not individual pieces
            
            # Pattern A: "RETURN source, type(old_var), old_var, labels(source) as source_labels, labels(old_var) as target_labels"
            # This is the most common pattern from LLM
            return_pattern = rf'RETURN\s+(\w+),\s+type\({old_var}\),\s+{old_var},\s+labels\(\1\)\s+as\s+source_labels,\s+labels\({old_var}\)\s+as\s+target_labels'
            if re.search(return_pattern, query):
                query = re.sub(
                    return_pattern,
                    r'RETURN \1, type(rel), target, labels(\1) as source_labels, labels(target) as target_labels',
                    query
                )
            else:
                # Fallback: Individual replacements (for unusual patterns)
                query = re.sub(rf'\blabels\({old_var}\)\s+as\s+target_labels\b', 'labels(target) as target_labels', query)
                query = re.sub(rf'\btype\({old_var}\)\b', 'type(rel)', query)
                # Replace remaining standalone old_var (but not inside function calls)
                # Use word boundary and check it's followed by comma or end
                query = re.sub(rf',\s+{old_var}\s*,', ', target,', query)
                query = re.sub(rf',\s+{old_var}\s+', ', target ', query)
        
        # 0b) Fix UNION queries with mismatched column names (CRITICAL FIX)
        if 'UNION' in query:
            # Pattern 1: N-way UNION with simple definition queries
            # MATCH (m:Label1) RETURN m, labels(m) UNION MATCH (s:Label2) RETURN s, labels(s) UNION ...
            # Problem: Column names are 'm' vs 's' vs 'mi' vs 'e' vs 'me' - Neo4j requires identical names
            # Fix: Use common column name like 'concept' and 'node_labels'
            
            # Extract all MATCH clauses
            parts = query.split('UNION')
            if len(parts) >= 2:
                # Pattern A: Simple definition UNION (no relationships)
                # MATCH (var:Label) RETURN var, labels(var) as node_labels
                simple_union_pattern = r'MATCH\s+\((\w+):(\w+)\s*\{[^}]*\}\)\s+RETURN\s+\1,\s+labels\(\1\)\s+as\s+node_labels'
                
                # Check if ALL parts match the simple pattern
                all_simple = all(re.search(simple_union_pattern, part.strip()) for part in parts)
                
                if all_simple:
                    # N-way simple definition UNION - standardize all to 'concept'
                    fixed_parts = []
                    var_names = []
                    
                    for part in parts:
                        part = part.strip()
                        match = re.search(simple_union_pattern, part)
                        if match:
                            var = match.group(1)
                            var_names.append(var)
                            # Replace: RETURN var, labels(var) → RETURN var as concept, labels(var) as node_labels
                            fixed_part = re.sub(
                                rf'RETURN\s+{var},\s+labels\({var}\)\s+as\s+node_labels',
                                f'RETURN {var} as concept, labels({var}) as node_labels',
                                part
                            )
                            fixed_parts.append(fixed_part)
                        else:
                            fixed_parts.append(part)
                    
                    if len(set(var_names)) > 1:  # Only log if variables differ
                        logger.info(f"Fixing {len(parts)}-way UNION column names: {', '.join(var_names)} → 'concept'")
                    
                    query = ' UNION '.join(fixed_parts)
                
                # Pattern B: Relationship UNION queries
                # MATCH (...)-[r:REL]->(...) RETURN ..., type(r), ..., labels(...) as source_labels, labels(...) as target_labels
                else:
                    relationship_union_pattern = r'MATCH\s+\([^)]+\)-\[r:[^\]]+\]->\([^)]+\)\s+RETURN\s+(\w+),\s+type\(r\),\s+(\w+),'
                    
                    # Check if ALL parts match the relationship pattern
                    all_relationship = all(re.search(relationship_union_pattern, part.strip()) for part in parts)
                    
                    if all_relationship:
                        # N-way relationship UNION - standardize to 'source', 'target'
                        fixed_parts = []
                        var_pairs = []
                        
                        for part in parts:
                            part = part.strip()
                            match = re.search(relationship_union_pattern, part)
                            if match:
                                var1 = match.group(1)  # source variable
                                var2 = match.group(2)  # target variable
                                var_pairs.append(f"{var1}/{var2}")
                                
                                # Replace: RETURN var1, type(r), var2, labels(var1) as source_labels, labels(var2) as target_labels
                                # → RETURN var1 as source, type(r) as rel_type, var2 as target, labels(var1) as source_labels, labels(var2) as target_labels
                                fixed_part = re.sub(
                                    rf'RETURN\s+{var1},\s+type\(r\),\s+{var2},\s+labels\({var1}\)\s+as\s+source_labels,\s+labels\({var2}\)\s+as\s+target_labels',
                                    f'RETURN {var1} as source, type(r) as rel_type, {var2} as target, labels({var1}) as source_labels, labels({var2}) as target_labels',
                                    part
                                )
                                fixed_parts.append(fixed_part)
                            else:
                                fixed_parts.append(part)
                        
                        if len(set(var_pairs)) > 1:  # Only log if variables differ
                            logger.info(f"Fixing {len(parts)}-way relationship UNION: {', '.join(var_pairs)} → 'source/target'")
                        
                        query = ' UNION '.join(fixed_parts)
                    
                    # Pattern C: Legacy - 2-way UNION with duplicate variables (keep for backward compatibility)
                    elif len(parts) == 2:
                        second_match = parts[1].strip()
                        dup_pattern = r'\((\w+):(\w+)[^)]*\)-\[r\]->\(\1:(\w+)'
                        match = re.search(dup_pattern, second_match)
                        if match:
                            dup_var = match.group(1)
                            logger.info(f"Fixing UNION duplicate variable: '{dup_var}'")
                            # Replace second occurrence of variable with unique name
                            second_match = re.sub(
                                rf'\({dup_var}:(\w+)([^)]*)\)\s+RETURN\s+{dup_var}\.name',
                                rf'(\1_2:\1\2) RETURN \1_2.name',
                                second_match
                            )
                            query = f"{parts[0].strip()} UNION {second_match}"
        
        # 1) Convert Cartesian product queries (comparison queries) to UNION
        # Pattern: MATCH (a:Label1), (b:Label2) RETURN ... 
        # This is inefficient and often returns 0 results due to cross-product
        if re.search(r'MATCH\s+\([^)]+\),\s*\([^)]+\)\s+RETURN', query):
            # Detect comparison patterns based on actual audit data
            comparison_pairs = [
                ('IntrinsicMotivation', 'ExtrinsicMotivation', 'Intrinsic', 'Extrinsic'),
                ('GrowthMindset', 'FixedMindset', 'Growth', 'Fixed'),
                ('PositiveStressEustress', 'NegativeStressDistress', 'Positive Stress', 'Negative Stress'),
                ('PositiveEmotions', 'NegativeEmotions', 'Positive Emotions', 'Negative Emotions'),
                ('HigherOrderThinking', 'LowerOrderThinking', 'Higher Order', 'Lower Order'),
                ('AdaptiveCoping', 'MaladaptiveCoping', 'Adaptive', 'Maladaptive'),
            ]
            
            for label1, label2, type1, type2 in comparison_pairs:
                # Pattern: MATCH (x:Label1 ...), (y:Label2 ...) RETURN ...
                pattern = rf'MATCH\s+\((\w+):{label1}[^)]*\),\s*\((\w+):{label2}[^)]*\)\s+RETURN\s+(.+?)(?:\s+LIMIT|$)'
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    var1 = match.group(1)
                    var2 = match.group(2)
                    return_clause = match.group(3)
                    
                    # Convert to UNION query with domain filter
                    union_query = (
                        f'MATCH ({var1}:{label1} {{domain: $domain}}) RETURN "{type1}" as type, {var1}.name as name, {var1}.description as description '
                        f'UNION MATCH ({var2}:{label2} {{domain: $domain}}) RETURN "{type2}" as type, {var2}.name as name, {var2}.description as description '
                        f'LIMIT 20'
                    )
                    logger.info(f"Converted Cartesian product to UNION query for {label1} vs {label2}")
                    return union_query
        
        # 2) Fix queries that look for non-existent or ambiguous generic labels
        # Based on actual 195 labels in audit report
        label_mappings = {
            # "Learning" is ambiguous - map to most common specific label
            r'\bLearning\b(?!Outcomes|Engagement|Development|Performance|Quality|Motivation|Strategies|Depth)': 'LearningOutcomes',
            # Fix plural/singular mismatches
            r'\bLearningProcesses\b': 'LearningProcess',  # ✅ FIX: Only singular exists
            r'\bProcesses\b(?!:)': 'Process',  # Generic plural fix
            # "Memory" exists as a single-node label, but context determines if we need WorkingMemory or LongTermMemory
            # Keep as is, since "Memory" is a valid label with 9 incoming relationships
            # "Emotions" exists as a label - keep as is
            # Map common abbreviations or variations to actual labels
            r'\bExecutive\sFunctions\b': 'ExecutiveFunctions',  # Handle space variation
            r'\bEmotional\sRegulation\b': 'EmotionalRegulation',
            r'\bWorking\sMemory\b': 'WorkingMemory',
            r'\bCritical\sThinking\b': 'CriticalThinking',
            # ✅ FIX: Mindset disambiguation
            r':Mindset\b(?!\w)': ':GrowthMindset',  # Default to GrowthMindset (more common in educational context)
        }
        
        for pattern, replacement in label_mappings.items():
            if re.search(pattern, query):
                query = re.sub(pattern, replacement, query)
                logger.info(f"Fixed label variation to canonical form: {replacement}")
        
        # 3) Ensure relationship patterns for "how" and "why" questions
        # These should use relationships, not standalone nodes
        if re.search(r'\b(how|why|does|affects?|influences?|impacts?|supports?|enhances?|leads?\sto)\b', query, re.IGNORECASE):
            # If query doesn't have a relationship pattern, log a warning
            if not re.search(r'-\[', query):
                logger.warning("Query asks 'how/why' but has no relationship pattern - may return incomplete results")
        
        # 4) Fix common Neuro relationship typos and variations
        # Based on actual 111 relationship types from audit
        relationship_fixes = {
            'SUPPORT': 'SUPPORTS',
            'AFFECT': 'AFFECTS',
            'ENHANCE': 'ENHANCES',  # Note: both ENHANCE and ENHANCES exist (29 vs 37 occurrences)
            'REDUCE': 'REDUCES',
            'FACILITATE': 'FACILITATES',
            'IMPAIR': 'IMPAIRS',
            'LEAD_TO': 'LEADS_TO',
            'INTERFERE': 'INTERFERE_WITH',
            'CONTRIBUTE': 'CONTRIBUTES_TO',
            'UNDERMINE': 'UNDERMINES',
        }
        
        for wrong, correct in relationship_fixes.items():
            query = re.sub(rf'-\[r:{wrong}\]->', f'-[r:{correct}]->', query)
        
        # 5) Handle plural/singular variations in actual data
        # Some labels exist in both forms (e.g., MemorySystem vs MemorySystems)
        # Prefer the plural form based on audit frequency
        plural_preferences = {
            'MemorySystem': 'MemorySystems',
            'AffectiveProcess': 'AffectiveProcesses',
            'CognitiveProcess': 'CognitiveProcesses',
        }
        
        for singular, plural in plural_preferences.items():
            query = re.sub(rf'\b{singular}\b', plural, query)
        
        return query
    
    def _repair_common_issues(self, query: str) -> str:
        """Apply common repairs that work across all domains"""
        import re
        
        # Fix newlines in string literals
        query = query.replace("\n", " ")
        
        # Fix missing closing parenthesis patterns
        query = re.sub(r'CONTAINS toLower\("([^"]+)"\s+RETURN', r'CONTAINS toLower("\1") RETURN', query)
        query = re.sub(r'CONTAINS toLower\("([^"]+)"\s+AND\s+', r'CONTAINS toLower("\1") AND ', query)
        query = re.sub(r'toLower\("([^"]+)"\s+RETURN', r'toLower("\1") RETURN', query)
        
        # Final hygiene: collapse duplicate spaces
        query = re.sub(r'\s+', ' ', query).strip()
        query = re.sub(r'WHERE\s+(AND|OR)\s+', 'WHERE ', query, flags=re.IGNORECASE)
        query = re.sub(r'\s+(AND|OR)\s+(RETURN|LIMIT)\b', r' \2', query, flags=re.IGNORECASE)
        
        return query
    
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
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, openai_api_key: str, model: str = "gpt-3.5-turbo-16k"):
        self.converter = Text2CypherConverter(neo4j_uri, neo4j_user, neo4j_password, openai_api_key, model)
    
    def process_question(self, question: str, domain: str = None, execute: bool = True) -> Dict:
        """Process a natural language question end-to-end with smart fallback
        
        Args:
            question: Natural language question
            domain: Domain filter ('udl', 'neuro', 'all', or None)
            execute: Whether to execute the query
        """
        # Convert to Cypher with domain filter
        cypher_query, metadata = self.converter.convert(question, domain=domain)
        
        result = {
            "question": question,
            "cypher_query": cypher_query,
            "metadata": metadata,
            "results": [],
            "execution_error": None,
            "used_fallback": False,
            "fallback_reason": None
        }
        
        # Execute if requested and query is valid
        if execute and metadata.get("is_valid", False):
            results, execution_error = self.converter.execute_query(cypher_query)
            
            # Smart Fallback System: Try relationship query first, fallback only on 0 results
            if not results and not execution_error:
                # Stage 1: Try more tolerant matching (existing logic)
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
                
                # Stage 2: If still no results and it's a relationship query, try definition fallback
                if not results and not execution_error and '-[' in cypher_query:
                    fallback_query = self._create_fallback_query(cypher_query, domain)
                    if fallback_query:
                        logger.info(f"[Fallback] Relationship query returned 0 results, trying definition fallback")
                        fallback_results, fallback_error = self.converter.execute_query(fallback_query)
                        
                        if fallback_results:
                            results = fallback_results
                            execution_error = fallback_error
                            result["used_fallback"] = True
                            result["fallback_query"] = fallback_query
                            result["fallback_reason"] = "relationship_missing"
                            logger.info(f"[Fallback] Definition fallback returned {len(results)} results")
            
            result["results"] = results
            result["execution_error"] = execution_error
        
        return result
    
    def _create_fallback_query(self, original_query: str, domain: str = None) -> Optional[str]:
        """Create a fallback definition query when relationship query fails
        
        Args:
            original_query: Original Cypher query that returned 0 results
            domain: Domain being queried
            
        Returns:
            Fallback query string or None if no fallback possible
        """
        import re
        
        # Extract the source node from relationship pattern: MATCH (source:Label)-[r]->...
        match = re.search(r'MATCH\s+\((\w+):(\w+)\s*(?:\{[^}]*\})?\)\s*-\[', original_query)
        if not match:
            return None
        
        var_name = match.group(1)
        label_name = match.group(2)
        
        # Create simple definition query for the source node
        if domain and domain != "all":
            fallback = f'MATCH ({var_name}:{label_name} {{domain: "{domain}"}}) RETURN {var_name}.name as name, {var_name}.category as category LIMIT 10'
        else:
            fallback = f'MATCH ({var_name}:{label_name}) RETURN {var_name}.name as name, {var_name}.category as category LIMIT 10'
        
        return fallback
    
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