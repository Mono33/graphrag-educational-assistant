#!/usr/bin/env python3
"""
llm_chain.py - Response Generator for Educational GraphRAG
Transforms structured EducationalContext into natural language responses for Italian teachers
"""

import logging
from typing import Dict, List, Optional
from dataclasses import asdict
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from context_builder import EducationalContext, ConfidenceLevel

logger = logging.getLogger(__name__)

class EducationalResponseGenerator:
    """
    Response Generator - Converts structured educational context into natural language
    Generates pedagogically sound, actionable responses for Italian teachers
    """
    
    def __init__(self, openai_api_key: str, language: str = "italian", temperature: float = 0.7):
        """
        Initialize the response generator
        
        Args:
            openai_api_key: OpenAI API key
            language: Target language for responses (default: italian)
            temperature: LLM temperature (0.7 for creative but coherent responses)
        """
        self.language = language
        self.llm = OpenAI(
            openai_api_key=openai_api_key,
            temperature=temperature,
            max_tokens=1500
        )
        
        # Load prompt templates and create chain using LCEL
        self.prompt_template = self._create_prompt_template()
        self.output_parser = StrOutputParser()
        self.chain = self.prompt_template | self.llm | self.output_parser
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create comprehensive prompt template for educational response generation"""
        
        if self.language == "italian":
            template = """Sei un esperto consulente pedagogico italiano specializzato in metodologie didattiche inclusive e differenziate.

Il tuo compito è fornire raccomandazioni chiare, pratiche e pedagogicamente solide per insegnanti italiani.

CONTESTO DELLA DOMANDA:
Domanda originale: {original_query}
Profilo studente: {student_profile}
Contesto educativo: {educational_context_type}

METODOLOGIE RACCOMANDATE:
{primary_methodologies}

METODOLOGIE DI SUPPORTO:
{supporting_methodologies}

EVIDENZA E BASI TEORICHE:
{evidence_summary}

PRIORITÀ DI IMPLEMENTAZIONE:
{implementation_priority}

LIVELLO DI CONFIDENZA:
{confidence_level}

STRATEGIE DI FALLBACK (se applicabili):
{fallback_strategies}

ISTRUZIONI PER LA RISPOSTA:

1. **Inizia con un'introduzione empatica** che riconosca la domanda dell'insegnante
2. **Presenta le metodologie principali** (massimo 3) in modo chiaro e strutturato:
   - Nome della metodologia
   - Perché è efficace per questo contesto specifico
   - Come implementarla in classe (passi concreti)
   - Adattamenti per bisogni speciali (se applicabile)
3. **Fornisci esempi pratici** per ogni metodologia
4. **Includi le basi teoriche** spiegando da dove provengono queste raccomandazioni
5. **Suggerisci un ordine di implementazione** con priorità chiare
6. **Aggiungi note sulla fiducia**: se la confidenza è bassa, suggerisci di consultare specialisti
7. **Usa un tono professionale ma accessibile**, evita il gergo eccessivo
8. **Formatta con elenchi puntati e sezioni chiare** per facilitare la lettura

IMPORTANTE:
- Rispondi SEMPRE in italiano
- Sii concreto e pratico, non teorico
- Fornisci azioni immediate che l'insegnante può prendere
- Adatta il linguaggio al contesto scolastico italiano (primaria, secondaria, etc.)
- Se la confidenza è BASSA o VERY_LOW, enfatizza la necessità di supporto specialistico

Genera la tua risposta pedagogica:"""
        
        else:  # English fallback
            template = """You are an expert educational consultant specializing in inclusive and differentiated teaching methodologies.

Your task is to provide clear, practical, and pedagogically sound recommendations for teachers.

QUERY CONTEXT:
Original question: {original_query}
Student profile: {student_profile}
Educational context: {educational_context_type}

RECOMMENDED METHODOLOGIES:
{primary_methodologies}

SUPPORTING METHODOLOGIES:
{supporting_methodologies}

EVIDENCE AND THEORETICAL BASIS:
{evidence_summary}

IMPLEMENTATION PRIORITY:
{implementation_priority}

CONFIDENCE LEVEL:
{confidence_level}

FALLBACK STRATEGIES (if applicable):
{fallback_strategies}

RESPONSE INSTRUCTIONS:

1. **Start with an empathetic introduction** acknowledging the teacher's question
2. **Present primary methodologies** (max 3) clearly and structured:
   - Methodology name
   - Why it's effective for this specific context
   - How to implement in classroom (concrete steps)
   - Adaptations for special needs (if applicable)
3. **Provide practical examples** for each methodology
4. **Include theoretical basis** explaining where recommendations come from
5. **Suggest implementation order** with clear priorities
6. **Add confidence notes**: if confidence is low, suggest consulting specialists
7. **Use professional but accessible tone**, avoid excessive jargon
8. **Format with bullet points and clear sections** for easy reading

IMPORTANT:
- Be concrete and practical, not theoretical
- Provide immediate actions the teacher can take
- If confidence is LOW or VERY_LOW, emphasize need for specialist support

Generate your pedagogical response:"""
        
        return PromptTemplate(
            input_variables=[
                "original_query",
                "student_profile",
                "educational_context_type",
                "primary_methodologies",
                "supporting_methodologies",
                "evidence_summary",
                "implementation_priority",
                "confidence_level",
                "fallback_strategies"
            ],
            template=template
        )
    
    async def generate_response(
        self, 
        educational_context: EducationalContext,
        original_query: str
    ) -> Dict[str, str]:
        """
        Generate natural language response from structured educational context
        
        Args:
            educational_context: Structured context from context_builder
            original_query: Original teacher's question
            
        Returns:
            Dict with:
                - response: Natural language response
                - confidence: Confidence level
                - metadata: Generation metadata
        """
        try:
            logger.info(f"Generating response for query: {original_query[:50]}...")
            
            # Prepare prompt inputs
            prompt_inputs = self._prepare_prompt_inputs(educational_context, original_query)
            
            # Generate response using LCEL chain
            response_text = self.chain.invoke(prompt_inputs)
            
            # Post-process response
            formatted_response = self._post_process_response(response_text, educational_context)
            
            return {
                'response': formatted_response,
                'confidence': educational_context.confidence_assessment.value,
                'metadata': {
                    'language': self.language,
                    'primary_methodologies_count': len(educational_context.primary_methodologies),
                    'evidence_sources': {
                        'total_nodes': educational_context.metadata.get('total_nodes', 0),
                        'total_triples': educational_context.metadata.get('total_triples', 0)
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self._generate_fallback_response(original_query, educational_context)
    
    def _prepare_prompt_inputs(
        self, 
        context: EducationalContext, 
        original_query: str
    ) -> Dict[str, str]:
        """Prepare inputs for the prompt template"""
        
        # Format student profile
        student_profile_text = self._format_student_profile(context.student_profile)
        
        # Format primary methodologies
        primary_methodologies_text = self._format_methodologies(
            context.primary_methodologies, 
            is_primary=True
        )
        
        # Format supporting methodologies
        supporting_methodologies_text = self._format_methodologies(
            context.supporting_methodologies, 
            is_primary=False
        )
        
        # Format implementation priority
        implementation_priority_text = "\n".join([
            f"  {i}. {priority}" 
            for i, priority in enumerate(context.implementation_priority, 1)
        ])
        
        # Format fallback strategies
        fallback_strategies_text = "\n".join([
            f"  - {strategy}" 
            for strategy in context.fallback_strategies
        ]) if context.fallback_strategies else "Nessuna strategia di fallback necessaria"
        
        return {
            'original_query': original_query,
            'student_profile': student_profile_text,
            'educational_context_type': context.student_profile.educational_context,
            'primary_methodologies': primary_methodologies_text,
            'supporting_methodologies': supporting_methodologies_text,
            'evidence_summary': context.evidence_summary,
            'implementation_priority': implementation_priority_text,
            'confidence_level': self._format_confidence_level(context.confidence_assessment),
            'fallback_strategies': fallback_strategies_text
        }
    
    def _format_student_profile(self, profile) -> str:
        """Format student profile for prompt"""
        if self.language == "italian":
            parts = []
            if profile.primary_needs:
                parts.append(f"Bisogni primari: {', '.join(profile.primary_needs)}")
            if profile.secondary_needs:
                parts.append(f"Bisogni secondari: {', '.join(profile.secondary_needs)}")
            parts.append(f"Contesto: {profile.educational_context}")
            if profile.grade_level:
                parts.append(f"Livello scolastico: {profile.grade_level}")
            if profile.subject_area:
                parts.append(f"Materia: {profile.subject_area}")
            return " | ".join(parts) if parts else "Profilo generale"
        else:
            parts = []
            if profile.primary_needs:
                parts.append(f"Primary needs: {', '.join(profile.primary_needs)}")
            if profile.secondary_needs:
                parts.append(f"Secondary needs: {', '.join(profile.secondary_needs)}")
            parts.append(f"Context: {profile.educational_context}")
            if profile.grade_level:
                parts.append(f"Grade level: {profile.grade_level}")
            if profile.subject_area:
                parts.append(f"Subject: {profile.subject_area}")
            return " | ".join(parts) if parts else "General profile"
    
    def _format_methodologies(self, methodologies: List, is_primary: bool = True) -> str:
        """Format methodology recommendations for prompt"""
        if not methodologies:
            return "Nessuna metodologia specifica identificata" if self.language == "italian" else "No specific methodologies identified"
        
        formatted = []
        for i, method in enumerate(methodologies, 1):
            method_text = f"\n{i}. **{method.name}** ({method.category})\n"
            method_text += f"   - Punteggio di rilevanza: {method.relevance_score:.2f}\n"
            method_text += f"   - Tipo di evidenza: {method.evidence_type}\n"
            method_text += f"   - Confidenza: {method.confidence.value}\n"
            method_text += f"   - Guida implementazione: {method.implementation_guidance}\n"
            
            if method.classroom_applications:
                method_text += f"   - Applicazioni in classe:\n"
                for app in method.classroom_applications[:3]:  # Top 3
                    method_text += f"     • {app}\n"
            
            if method.special_considerations:
                method_text += f"   - Considerazioni speciali:\n"
                for consideration in method.special_considerations[:3]:  # Top 3
                    method_text += f"     • {consideration}\n"
            
            formatted.append(method_text)
        
        return "\n".join(formatted)
    
    def _format_confidence_level(self, confidence: ConfidenceLevel) -> str:
        """Format confidence level with explanation"""
        confidence_map_it = {
            ConfidenceLevel.VERY_HIGH: "MOLTO ALTA - Raccomandazioni basate su evidenze dirette e robuste",
            ConfidenceLevel.HIGH: "ALTA - Raccomandazioni supportate da buone evidenze",
            ConfidenceLevel.MEDIUM: "MEDIA - Raccomandazioni ragionevoli, considerare supporto aggiuntivo",
            ConfidenceLevel.LOW: "BASSA - Consultare specialisti per raccomandazioni personalizzate",
            ConfidenceLevel.VERY_LOW: "MOLTO BASSA - Necessario supporto specialistico"
        }
        
        confidence_map_en = {
            ConfidenceLevel.VERY_HIGH: "VERY HIGH - Recommendations based on direct and robust evidence",
            ConfidenceLevel.HIGH: "HIGH - Recommendations supported by good evidence",
            ConfidenceLevel.MEDIUM: "MEDIUM - Reasonable recommendations, consider additional support",
            ConfidenceLevel.LOW: "LOW - Consult specialists for personalized recommendations",
            ConfidenceLevel.VERY_LOW: "VERY LOW - Specialist support required"
        }
        
        if self.language == "italian":
            return confidence_map_it.get(confidence, "SCONOSCIUTA")
        else:
            return confidence_map_en.get(confidence, "UNKNOWN")
    
    def _post_process_response(self, response_text: str, context: EducationalContext) -> str:
        """Post-process and enhance the generated response"""
        
        # Add confidence warning if needed
        if context.confidence_assessment in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW]:
            if self.language == "italian":
                warning = "\n\n⚠️ **NOTA IMPORTANTE**: Queste raccomandazioni hanno un livello di confidenza basso. Ti consiglio vivamente di consultare un pedagogista o uno specialista per un supporto personalizzato.\n"
            else:
                warning = "\n\n⚠️ **IMPORTANT NOTE**: These recommendations have a low confidence level. We strongly recommend consulting a pedagogist or specialist for personalized support.\n"
            response_text = warning + response_text
        
        # Add evidence footer
        if self.language == "italian":
            footer = f"\n\n---\n📊 **Fonti**: Questa risposta si basa su {context.metadata.get('total_nodes', 0)} concetti educativi e {context.metadata.get('total_triples', 0)} relazioni pedagogiche dal grafo della conoscenza.\n"
        else:
            footer = f"\n\n---\n📊 **Sources**: This response is based on {context.metadata.get('total_nodes', 0)} educational concepts and {context.metadata.get('total_triples', 0)} pedagogical relationships from the knowledge graph.\n"
        
        response_text += footer
        
        return response_text.strip()
    
    def _generate_fallback_response(self, query: str, context: EducationalContext) -> Dict[str, str]:
        """Generate fallback response when main generation fails"""
        
        if self.language == "italian":
            fallback_text = f"""Mi dispiace, ho riscontrato un problema nella generazione della risposta completa.

Tuttavia, ecco alcune raccomandazioni di base basate sul contesto educativo:

**Metodologie Suggerite:**
{self._format_methodologies(context.primary_methodologies, is_primary=True)}

**Prossimi Passi:**
{chr(10).join([f'{i}. {p}' for i, p in enumerate(context.implementation_priority[:3], 1)])}

Per un supporto più dettagliato, ti consiglio di:
- Consultare un pedagogista specializzato
- Rivedere le linee guida UDL (Universal Design for Learning)
- Collaborare con il team di supporto educativo della tua scuola

Se hai bisogno di ulteriore assistenza, non esitare a chiedere!"""
        else:
            fallback_text = f"""I apologize, I encountered an issue generating the complete response.

However, here are some basic recommendations based on the educational context:

**Suggested Methodologies:**
{self._format_methodologies(context.primary_methodologies, is_primary=True)}

**Next Steps:**
{chr(10).join([f'{i}. {p}' for i, p in enumerate(context.implementation_priority[:3], 1)])}

For more detailed support, I recommend:
- Consult a specialized pedagogist
- Review UDL (Universal Design for Learning) guidelines
- Collaborate with your school's educational support team

If you need further assistance, don't hesitate to ask!"""
        
        return {
            'response': fallback_text,
            'confidence': context.confidence_assessment.value,
            'metadata': {
                'language': self.language,
                'fallback': True,
                'error': 'LLM generation failed'
            }
        }

# Integration helper for testing
async def test_response_generator():
    """Test the response generator with sample educational context"""
    import asyncio
    from context_builder import (
        EducationalContextBuilder,
        MethodologyRecommendation,
        StudentProfile,
        EducationalContext,
        ConfidenceLevel
    )
    from config import config
    
    # Create sample educational context
    sample_context = EducationalContext(
        student_profile=StudentProfile(
            primary_needs=["visual_impairment", "tactile_learning"],
            secondary_needs=["audio_support"],
            educational_context="special_needs",
            grade_level="quarta elementare",
            subject_area="scienze"
        ),
        primary_methodologies=[
            MethodologyRecommendation(
                name="Cooperative Learning",
                category="Collaborative Pedagogy",
                relevance_score=0.92,
                evidence_type="direct_relationship",
                implementation_guidance="Organize students in diverse groups of 3-5 members",
                classroom_applications=[
                    "Jigsaw method for complex topics",
                    "Think-Pair-Share for quick engagement",
                    "Group investigations for project work"
                ],
                special_considerations=[
                    "Provide tactile materials for visually impaired students",
                    "Use verbal descriptions during group activities",
                    "Assign complementary roles based on abilities"
                ],
                confidence=ConfidenceLevel.VERY_HIGH
            ),
            MethodologyRecommendation(
                name="Station Rotation",
                category="Differentiated Instruction",
                relevance_score=0.85,
                evidence_type="semantic_similarity",
                implementation_guidance="Multiple learning stations with different activities",
                classroom_applications=[
                    "Tactile learning station with 3D models",
                    "Audio description station",
                    "Peer support station"
                ],
                special_considerations=[
                    "Include braille materials where possible",
                    "Use high-contrast visual aids",
                    "Ensure safe navigation between stations"
                ],
                confidence=ConfidenceLevel.HIGH
            )
        ],
        supporting_methodologies=[],
        evidence_summary="Direct pedagogical evidence: 2 direct methodology suggestions | Conceptual similarity: Average conceptual similarity of 0.78 across 5 related concepts",
        implementation_priority=[
            "Start with Cooperative Learning (high confidence)",
            "Ensure accessibility accommodations are in place",
            "Begin with small-group implementation",
            "Pilot with a subset of students first"
        ],
        confidence_assessment=ConfidenceLevel.VERY_HIGH,
        fallback_strategies=[],
        metadata={
            'total_nodes': 15,
            'total_triples': 6,
            'semantic_nodes': 8,
            'graph_nodes': 7,
            'original_query': "Quali metodologie funzionano meglio per studenti con deficit di attenzione?",
            'query_type': 'special_needs'
        }
    )
    
    # Initialize response generator
    generator = EducationalResponseGenerator(
        openai_api_key=config.openai.api_key,
        language="italian"
    )
    
    # Generate response
    original_query = "Come posso adattare una lezione di scienze per studenti ipovedenti in quarta elementare?"
    
    print("="*80)
    print("🎓 EDUCATIONAL RESPONSE GENERATOR TEST")
    print("="*80)
    print(f"\n📝 Query: {original_query}")
    print(f"\n🔄 Generating response...")
    
    result = await generator.generate_response(sample_context, original_query)
    
    print(f"\n✅ Response generated!")
    print(f"📊 Confidence: {result['confidence']}")
    print(f"🌍 Language: {result['metadata']['language']}")
    print(f"\n" + "="*80)
    print("📖 GENERATED RESPONSE:")
    print("="*80)
    print(result['response'])
    print("="*80)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_response_generator())

