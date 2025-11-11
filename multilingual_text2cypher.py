#!/usr/bin/env python3
"""
multilingual_text2cypher.py - Production multilingual text2cypher for Italian teachers
Enhanced with comprehensive educational terminology including UDL, BES, DSA concepts
"""

import re
from typing import Dict, List, Tuple, Optional
from text2cypher import Text2CypherPipeline
from config import config
import logging

logger = logging.getLogger(__name__)

class MultilingualText2Cypher:
    """Production-ready multilingual text2cypher for Italian educational queries"""
    
    def __init__(self):
        # Use config.py which loads from .env
        self.pipeline = Text2CypherPipeline(
            config.neo4j.uri,
            config.neo4j.user,
            config.neo4j.password,
            config.openai.api_key,
            config.openai.model  # Uses model from .env (e.g., gpt-4o)
        )
        
        # UDL-specific Italian → English educational term mapping
        # Mapped to ACTUAL node names in your Neo4j database
        self.udl_terms = {
            # Special Educational Needs - Map to actual StudentWithSpecialNeeds nodes
            "ipovedenti": "Blind",  # ✅ Exists in your data
            "disabilità uditive": "Deaf",  # ✅ Exists in your data  
            "disabilità": "Physical disability",  # ✅ Exists in your data
            "disabilità fisica": "Physical disability",  # ✅ Exact match
            "dislessia": "Language difficulties due to foreign origin",  # ✅ Closest match
            "difficoltà linguistiche": "Language difficulties due to foreign origin",  # ✅ CRITICAL: Direct mapping for linguistic difficulties
            "ADHD": "Adhd",  # ✅ Exists (case sensitive)
            "deficit di attenzione": "Attention Deficit",  # ✅ Exists
            "autismo": "Autism spectrum disorder",  # ✅ Exists
            "disturbi dello spettro autistico": "Autism spectrum disorder",  # ✅ CRITICAL FIX - Added!
            "motivazione": "NoPersonalMotivation",  # ✅ Exists (for lack of motivation)
            "senza motivazione": "NoPersonalMotivation",  # ✅ Alternative phrasing
            "eccellenza": "Excellence in some or all subjects",  # ✅ Exists
            "difficoltà cognitive": "Cognitive disability [mild, moderate, severe]",  # ✅ Exact match with brackets
            "difficoltà cognitive lievi": "Cognitive disability [mild, moderate, severe]",  # ✅ Exact match
            "difficoltà cognitive moderate": "Cognitive disability [mild, moderate, severe]",  # ✅ Exact match
            "difficoltà cognitive gravi": "Cognitive disability [mild, moderate, severe]",  # ✅ Exact match
            "difficoltà di lettura": "reading difficulties",  # Keep generic term
            "iperattività": "Hyperactivity Disorder",  # ✅ Exists in your data
            "disturbo oppositivo": "Oppositional Defiant Disorder - ODD",  # ✅ Exists in your data
            "plusdotazione": "Giftedness",  # ✅ Exists in your data
            "muto": "Mute or no verbal",  # ✅ Exists in your data
            # Removed non-existent terms: discalculici, discalculia, dislessici, BES, DSA, etc.
            
            # UDL (Universal Design for Learning)
            "UDL": "Universal Design for Learning",
            "Universal Design for Learning": "Universal Design for Learning",
            "progettazione universale": "universal design",
            "linee guida UDL": "UDL guidelines",
            "principi UDL": "UDL principles",
            "strategie UDL": "UDL strategies",
            "tecniche UDL": "UDL techniques",
            "framework UDL": "UDL framework",
            
            # Teaching and Learning Methods - Map to actual PedagogicalMethodology nodes
            "apprendimento cooperativo": "Cooperative Learning",  # ✅ Exists in your data
            "flipped classroom": "Flipped Classroom",  # ✅ Exists in your data
            "game based learning": "GameBasedLearning",  # ✅ Exists in your data
            "debate": "Debate",  # ✅ Exists in your data
            "project based learning": "Project based learning",  # ✅ Exists in your data
            "role playing": "Role Playing, Debate",  # ✅ Exists in your data
            "station rotation": "Station Rotation",  # ✅ Exists in your data
            "stem": "Stem",  # ✅ Exists in your data
            "peertopeereducation": "Peertopeereducation",  # ✅ Exists in your data
            
            # Teaching Approaches - Map to actual TeachingApproach nodes
            "lezioni frontali": "Frontal lessons",  # ✅ Exists in your data
            "lezioni frontali lunghe": "long frontal lessons",  # ✅ Exists in your data
            "supporti visivi, alternative bilingue": "Visual supports, bilingual alternatives",  # ✅ Exists in your data
            
            # Generic terms (keep as they are useful for query context)
            "obiettivi diversificati": "differentiated objectives",
            "obiettivi didattici": "learning objectives",
            "classe eterogenea": "heterogeneous class",
            "classe multilingue": "multilingual class",
            "risorse visive": "visual resources",
            "risorse analogiche": "analog resources",
            "supporti visivi": "visual supports",
            "supporti linguistici": "linguistic supports",
            "supporti motori": "motor supports",
            "attività": "activities",
            "partecipazione": "participation",
            "prerequisiti": "prerequisites",
            "metodologie": "methodologies",
            "strategie": "strategies",
            "strumenti tecnologici": "technological tools",
            "unità didattica": "didactic unit",
            "sequenza didattica": "didactic sequence",
            
            # Assessment and Evaluation  
            "valutazione formativa": "formative assessment",
            "valutazione": "assessment",
            "verifiche": "tests",
            "competenze linguistiche": "linguistic competencies",
            "comprensione del testo": "text comprehension",
            "comprensione della lettura": "reading comprehension",
            "DSA": "Language difficulties due to foreign origin",  # ✅ Map DSA to closest existing term  (check that 14/09/25)
            "durante le verifiche": "during tests",  # For assessment context
            
            # Subject Areas
            "scienze": "science",
            "storia": "history", 
            "matematica": "mathematics",
            "geografia": "geography",
            "inglese": "English",
            "lingua inglese": "English language",
            "frazioni": "fractions",
            "biodiversità": "biodiversity",
            "prima guerra mondiale": "World War I",
            
            # Grade Levels
            "quarta elementare": "fourth grade",
            "prima elementare": "first grade",
            "seconda elementare": "second grade",
            "terza elementare": "third grade",
            "quinta elementare": "fifth grade",
            "3 elementare": "third grade",
            "prima media": "sixth grade",
            "prima superiore": "ninth grade",
            
            # Learning Concepts
            "apprendimento": "learning",
            "motivazione": "motivation",
            "autonomia": "autonomy",
            "inclusiva": "inclusive",
            "accessibile": "accessible",
            "adattamenti": "adaptations",
            "differenziare": "differentiate",
            "modalità alternative": "alternative methods",
            "rappresentazione dei contenuti": "content representation",
            
            # Environmental Factors - Map to actual node labels
            "illuminazione": "Lighting",  # ✅ Exists as node label
            "colori": "Colour",  # ✅ Exists as node label
            "acustica": "Acoustic",  # ✅ Exists as node label
            "arredi": "Furniture",  # ✅ Exists as node label
            "texture": "Texture",  # ✅ Exists as node label
            "odori": "Smell",  # ✅ Exists as node label
            "ambiente di apprendimento": "LearningEnvironment",  # ✅ Exists as node label
            "barriere ambientali": "EnvironmentalBarrier",  # ✅ Exists as node label
            "supporto ambientale": "EnvironmentalSupport",  # ✅ Exists as node label
            "rischi ambientali": "EnvironmentalRisk",  # ✅ Exists as node label
            "strategie ambientali": "EnvironmentalStrategy",  # ✅ Exists as node label
            "infrastruttura": "Infrastructure",  # ✅ Exists as node label
            "barriere infrastrutturali": "InfrastructureBarrier",  # ✅ Exists as node label
            "lavagna interattiva": "InteractiveBoard",  # ✅ Exists as node label
            
            # Student Characteristics - Map to actual StudentCharacteristic nodes
            "eccellenza in alcune o tutte le materie": "Excellence in some or all subjects",  # ✅ Exists
            "difficoltà in alcune o tutte le materie": "Struggles in some or all subjects",  # ✅ Exists
            "alta motivazione intrinseca": "High intrinsic motivation",  # ✅ Exists
            "mancanza di motivazione": "Lack of motivation",  # ✅ Exists
            "timidezza significativa": "Significant shyness or relational closure",  # ✅ Exists
            "problemi personali o familiari": "Personal or family issues beyond our control",  # ✅ Exists
            
            # Context - Map to actual Context nodes
            "coesivo": "Cohesive",  # ✅ Exists
            "diviso in gruppi": "Split in groups",  # ✅ Exists
            "con elementi disturbanti": "With disruptive elements",  # ✅ Exists
            "motivato": "Motivated",  # ✅ Exists
            "divario di genere": "Gender gap",  # ✅ Exists
            
            # General Terms (keep as they are useful for query context)
            "studenti": "students",
            "ragazzi": "students",
            "bambini": "children",
            "insegnare": "teach",
            "adattare": "adapt",
            "aiutare": "help",
            "favorire": "promote",
            "facilitare": "facilitate",
            "progettare": "design",
            "integrare": "integrate",
            "utilizzare": "use",
            "supportare": "support",
            "raggiungere": "achieve",
            "migliorare": "improve"
        }
        
        # Neuro-specific Italian → English term mapping
        self.neuro_terms = {
            # Neuroscience Core Concepts
            "attenzione": "attention",
            "attenzione selettiva": "selective attention",
            "attenzione divisa": "divided attention",
            "attenzione sostenuta": "sustained attention",
            
            # Memory Systems
            "memoria": "memory",
            "memoria di lavoro": "working memory",
            "memoria a lungo termine": "long term memory",
            "memoria a breve termine": "short term memory",
            "consolidamento": "consolidation",
            
            # Executive Functions
            "funzioni esecutive": "executive functions",
            "controllo esecutivo": "executive control",
            "controllo cognitivo": "cognitive control",
            "controllo inibitorio": "inhibitory control",
            "flessibilità cognitiva": "cognitive flexibility",
            
            # Thinking & Cognition
            "pensiero critico": "critical thinking",
            "creatività": "creativity",
            "pensiero divergente": "divergent thinking",
            "risoluzione problemi": "problem solving",
            "ragionamento": "reasoning",
            
            # Emotions & Motivation
            "emozioni": "emotions",
            "emozioni positive": "positive emotions",
            "emozioni negative": "negative emotions",
            "motivazione": "motivation",
            "motivazione intrinseca": "intrinsic motivation",
            "motivazione estrinseca": "extrinsic motivation",
            "regolazione emotiva": "emotional regulation",
            
            # Learning Processes
            "apprendimento": "learning",
            "carico cognitivo": "cognitive load",
            "risultati apprendimento": "learning outcomes",
            "processi cognitivi": "cognitive processes",
            "elaborazione informazioni": "information processing",
            
            # Stress & Mindset
            "stress": "stress",
            "stress positivo": "positive stress",
            "stress negativo": "negative stress",
            "mentalità crescita": "growth mindset",
            "mentalità fissa": "fixed mindset",
            "resilienza": "resilience",
            
            # Metacognition
            "metacognizione": "metacognition",
            "autoregolazione": "self regulation",
            "consapevolezza": "awareness",
            "monitoraggio": "monitoring",
            "valutazione": "evaluation",
            
            # Social & Communication
            "cognizione sociale": "social cognition",
            "apprendimento sociale": "social learning",
            "comunicazione": "communication",
            "comprensione sociale": "social understanding",
            
            # General Terms
            "cervello": "brain",
            "neurale": "neural",
            "cognitivo": "cognitive",
            "affettivo": "affective",
            "comportamentale": "behavioral"
        }
        
        # Advanced query patterns for complex educational concepts
        self.query_patterns = {
            "udl_queries": [
                "UDL", "universal design", "linee guida", "principi", "framework"
            ],
            "assessment_queries": [
                "valutazione", "verifiche", "competenze", "comprensione"
            ],
            "special_needs_queries": [
                "BES", "DSA", "disabilità", "difficoltà", "dislessia", "discalculia", "ipovedenti"
            ],
            "differentiation_queries": [
                "diversificat", "eterogenea", "adattamenti", "inclusiv", "accessibil"
            ]
        }
    
    def detect_language(self, query: str) -> str:
        """Enhanced language detection for educational queries"""
        italian_indicators = [
            # Basic indicators
            "come", "cosa", "quali", "che", "per", "con", "gli", "delle", "nella", 
            "posso", "sono", "può", "hanno", "studenti", "bambini", "lezione",
            # Educational indicators
            "obiettivi", "classe", "metodologie", "strategie", "valutazione",
            "apprendimento", "insegnare", "adattare", "facilitare", "supportare",
            # Neuroscience indicators (NEW!)
            "mentalità", "crescita", "differenza", "significa", "stress", 
            "positivo", "negativo", "motivazione", "intrinseca", "estrinseca"
        ]
        
        query_lower = query.lower()
        italian_count = sum(1 for word in italian_indicators if word in query_lower)
        
        # Lower threshold to 1 for better detection (was 2)
        return "italian" if italian_count >= 1 else "english"
    
    def detect_query_type(self, query: str) -> List[str]:
        """Detect the type of educational query for better processing"""
        query_lower = query.lower()
        detected_types = []
        
        for pattern_type, keywords in self.query_patterns.items():
            if any(keyword.lower() in query_lower for keyword in keywords):
                detected_types.append(pattern_type)
        
        return detected_types
    
    def _calculate_translation_coverage(self, original: str, translated: str) -> float:
        """
        Calculate how much of the original Italian query was successfully translated.
        
        Args:
            original: Original Italian query
            translated: Translated query (may be partial)
            
        Returns:
            Coverage percentage (0.0 to 1.0)
        """
        # Extract meaningful words (ignore stopwords and short words)
        italian_stopwords = {'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una', 'di', 'a', 'da', 
                            'in', 'con', 'su', 'per', 'tra', 'fra', 'e', 'o', 'ma', 'se', 'che', 'chi'}
        
        original_words = set(w.lower() for w in original.split() if len(w) > 2 and w.lower() not in italian_stopwords)
        translated_words = set(w.lower() for w in translated.split() if len(w) > 2 and w.lower() not in italian_stopwords)
        
        if not original_words:
            return 1.0
        
        # Count how many Italian words still remain in translated text (= not translated)
        untranslated_count = sum(1 for word in original_words if word in translated_words)
        
        # Coverage = percentage of words that WERE translated
        coverage = 1.0 - (untranslated_count / len(original_words))
        
        return coverage
    
    def _translate_with_openai(self, italian_query: str, domain: str) -> str:
        """
        Fallback translation using OpenAI API (for queries with low dictionary coverage).
        
        This ensures 100% coverage for any Italian query, even with uncommon terms.
        Cost: ~$0.0001 per query (gpt-4o-mini)
        
        Args:
            italian_query: Original Italian query
            domain: Domain context ('udl', 'neuro', 'all')
            
        Returns:
            Fully translated English query
        """
        try:
            from openai import OpenAI
            import os
            
            # Initialize OpenAI client
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("OPENAI_API_KEY not found in environment")
                return italian_query
            
            client = OpenAI(api_key=api_key)
            
            # Set context based on domain
            if domain == "neuro":
                context = "neuroscience and cognitive science"
            elif domain == "udl":
                context = "education and Universal Design for Learning"
            else:
                context = "education"
            
            # Translation prompt
            prompt = f"""Translate this Italian query to English. Context: {context}.
Keep technical terms accurate and preserve the meaning.

Italian: {italian_query}
English:"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Fast and cheap
                messages=[{"role": "user", "content": prompt}],
                temperature=0,  # Deterministic
                max_tokens=150
            )
            
            translated = response.choices[0].message.content.strip()
            
            # Remove any quotes that OpenAI might add
            translated = translated.strip('"\'')
            
            logger.info(f"[OpenAI Translation] {italian_query[:50]}... → {translated[:50]}...")
            
            return translated
            
        except Exception as e:
            logger.error(f"OpenAI translation failed: {e}")
            return italian_query  # Fallback to original
    
    def enhance_italian_query(self, italian_query: str, domain: str = None) -> str:
        """
        Enhanced Italian query processing with HYBRID translation approach.
        
        HYBRID STRATEGY:
        1. Try dictionary-based translation (fast, free)
        2. Calculate translation coverage
        3. If coverage < 50%, use OpenAI translation (accurate, small cost)
        
        This ensures 100% coverage while minimizing API costs.
        
        Args:
            italian_query: Italian language query
            domain: Domain filter ('udl', 'neuro', 'all', or None)
        """
        enhanced_query = italian_query
        
        # Select appropriate term dictionary based on domain
        if domain == "udl":
            italian_terms = self.udl_terms
        elif domain == "neuro":
            # For Neuro domain, ONLY use neuroscience terms
            # Filter out UDL-triggering pedagogical terms
            italian_terms = self.neuro_terms.copy()
            
            # Remove terms that trigger UDL patterns in Neuro domain
            udl_trigger_terms = {
                "strategie": "",  # Don't translate - keeps query focused on concepts
                "metodologie": "",  # Don't translate
                "attività": "",  # Don't translate
                "verifiche": "",  # Don't translate (avoid Assessment label)
                "valutazione": "",  # Don't translate
            }
            
            # Keep only neuroscience terms, remove UDL triggers
            for term in udl_trigger_terms:
                if term in italian_terms:
                    del italian_terms[term]
                    
        elif domain == "all" or domain is None:
            # Merge both dictionaries for cross-domain or unspecified
            italian_terms = {**self.udl_terms, **self.neuro_terms}
        else:
            italian_terms = self.udl_terms  # Default to UDL
        
        # STEP 1: Dictionary-based translation (fast, free)
        sorted_terms = sorted(italian_terms.items(), key=lambda x: len(x[0]), reverse=True)
        
        for italian_term, english_term in sorted_terms:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(italian_term), re.IGNORECASE)
            enhanced_query = pattern.sub(english_term, enhanced_query)
        
        # STEP 2: Calculate translation coverage
        coverage = self._calculate_translation_coverage(italian_query, enhanced_query)
        
        logger.info(f"[Translation] Dictionary coverage: {coverage:.0%}")
        
        # STEP 3: If coverage is low, use OpenAI fallback
        if coverage < 0.5:
            logger.warning(f"[Translation] Coverage {coverage:.0%} < 50%, using OpenAI fallback")
            enhanced_query = self._translate_with_openai(italian_query, domain)
        else:
            logger.info(f"[Translation] Coverage {coverage:.0%} >= 50%, using dictionary translation")
        
        # Detect query types for additional context
        query_types = self.detect_query_type(italian_query)
        
        # Add educational context based on query type
        context_parts = ["Educational query"]
        
        # For Neuro domain, use neuroscience-specific context (not pedagogical)
        if domain == "neuro":
            context_parts = ["Neuroscience query"]
        elif "udl_queries" in query_types:
            context_parts.append("Universal Design for Learning context")
        
        if "special_needs_queries" in query_types and domain != "neuro":
            context_parts.append("Special Educational Needs context")
        
        if "assessment_queries" in query_types and domain != "neuro":
            context_parts.append("Assessment and evaluation context")
        
        if "differentiation_queries" in query_types and domain != "neuro":
            context_parts.append("Differentiated instruction context")
        
        context_prefix = f"{': '.join(context_parts)}: "
        enhanced_query = context_prefix + enhanced_query
        
        return enhanced_query
    
    def process_query(self, query: str, domain: str = None, execute: bool = True) -> Dict:
        """Process query with enhanced multilingual and educational support
        
        Args:
            query: Natural language query in Italian or English
            domain: Domain filter ('udl', 'neuro', 'all', or None)
            execute: Whether to execute the query
        """
        original_query = query
        language = self.detect_language(query)
        query_types = self.detect_query_type(query)
        
        logger.info(f"Processing {language} query (domain={domain}) with types {query_types}: {query[:50]}...")
        
        # Enhance Italian queries for better Neo4j mapping with domain-specific terms
        if language == "italian":
            enhanced_query = self.enhance_italian_query(query, domain=domain)
            logger.info(f"Enhanced query: {enhanced_query}")
        else:
            enhanced_query = query
        
        # Process with text2cypher pipeline (now with domain support)
        result = self.pipeline.process_question(enhanced_query, domain=domain, execute=execute)
        
        # Add comprehensive multilingual metadata
        result.update({
            "original_query": original_query,
            "detected_language": language,
            "detected_query_types": query_types,
            "enhanced_query": enhanced_query if language == "italian" else None,
            "multilingual_processing": True,
            "educational_context": True
        })
        
        return result
    
    def batch_process_queries(self, queries: List[str], execute: bool = False) -> List[Dict]:
        """Process multiple queries efficiently with detailed reporting"""
        results = []
        
        # Statistics tracking
        language_stats = {"italian": 0, "english": 0}
        type_stats = {}
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}")
            result = self.process_query(query, execute=execute)
            results.append(result)
            
            # Update statistics
            lang = result.get('detected_language', 'unknown')
            language_stats[lang] = language_stats.get(lang, 0) + 1
            
            for query_type in result.get('detected_query_types', []):
                type_stats[query_type] = type_stats.get(query_type, 0) + 1
        
        # Log statistics
        logger.info(f"Batch processing complete:")
        logger.info(f"  Language distribution: {language_stats}")
        logger.info(f"  Query type distribution: {type_stats}")
        
        return results
    
    def close(self):
        """Close pipeline connections"""
        self.pipeline.close()

# Enhanced testing function for comprehensive teacher queries
def test_comprehensive_italian_queries():
    """Test function for comprehensive Italian teacher queries including UDL, BES, DSA"""
    
    # Validate configuration first
    is_valid, errors = config.validate()
    if not is_valid:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        print("\n💡 Update your .env file with:")
        print("NEO4J_PASSWORD=your_actual_password")
        print("OPENAI_API_KEY=sk-your-actual-key")
        return
    
    # Comprehensive test queries from teachers_queries.txt
    test_queries = [
        # UDL queries
        "Quali strategie UDL posso integrare in una unità didattica sulla biodiversità?",
        "Come posso aiutare gli studenti a raggiungere degli obiettivi seguendo le linee guida dell'UDL?",
        "Esistono metodologie che possano essere utilizzate insieme al framework dell'UDL?",
        
        # Special needs queries  
        "Come posso adattare una lezione di scienze per studenti con disabilità uditive?",
        "Quali prerequisiti sono necessari per insegnare le frazioni con bambini discalculici?",
        "Ci sono strategie per i ragazzi ipovedenti?",
        "Quali risorse posso utilizzare per supportare studenti con DSA durante le verifiche?",
        
        # Assessment queries
        "Esistono esempi di valutazione formativa di lingua inglese per studenti con dislessia?",
        "Come posso valutare in modo formativo le competenze linguistiche in una classe multilingue?",
        
        # Differentiation queries
        "Come posso determinare obiettivi diversificati all'interno di una classe eterogenea?",
        "Come posso differenziare una lezione di geografia per studenti con livelli diversi?",
        
        # Technology integration
        "Quali strumenti tecnologici posso usare per facilitare l'apprendimento di inglese con studenti dislessici?",
        "Come posso progettare una lezione di matematica che includa supporti visivi usando l'AI?"
    ]
    
    multilingual_processor = MultilingualText2Cypher()
    
    try:
        print("🚀 Testing Comprehensive Italian Teacher Queries")
        print("🎓 Including UDL, BES, DSA, Assessment, and Technology Integration")
        print("=" * 100)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🇮🇹 TEST {i}: {query}")
            print("-" * 80)
            
            result = multilingual_processor.process_query(query, execute=False)
            
            print(f"Language: {result['detected_language']}")
            print(f"Query Types: {', '.join(result.get('detected_query_types', ['general']))}")
            print(f"Enhanced: {result.get('enhanced_query', 'N/A')[:100]}...")
            print(f"Cypher: {result['cypher_query']}")
            print(f"Valid: {result['metadata'].get('is_valid', False)}")
            
            if result['metadata'].get('validation_error'):
                print(f"Error: {result['metadata']['validation_error']}")
    
    finally:
        multilingual_processor.close()

if __name__ == "__main__":
    test_comprehensive_italian_queries() 