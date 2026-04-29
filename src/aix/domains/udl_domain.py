#!/usr/bin/env python3
"""
UDL Domain Configuration — Updated March 2026
All UDL-specific configuration based on the NEW 763-node KG
(KG_Data_Collection_UDL_03206.xlsx → cleaned → ingested)

Graph summary (March 2026):
  763 nodes  |  271 unique labels  |  799 relationships  |  35 rel types
  Top labels: Checkpoint(55), AnalogicalTool(36), DigitalTool(29),
              EducationalApproach(26), BehavioralManifestations(26),
              InstructionalStrategy(21), LearningChallenge(20), ...
  Top rels:   MITIGATED_BY(178), SUGGESTS(148), ASSOCIATES_TO(75),
              SUPPORTS_BY(73), MENTIONS(55), LEADS(54), CAUSES(50), ...
"""

from aix.domains.base_config import BaseDomainConfig
from typing import Dict, List, Tuple
import re


class UDLDomainConfig(BaseDomainConfig):
    """UDL (Universal Design for Learning) domain configuration"""

    # ============================================================
    # CORE DOMAIN IDENTITY
    # ============================================================

    def _get_name(self) -> str:
        return "udl"

    def _get_display_name(self) -> str:
        return "UDL (Universal Design for Learning)"

    def _get_icon(self) -> str:
        return "🎯"

    # ============================================================
    # NODE2VEC WEIGHTS
    # Formula: 1.0 + log10(nodes)*0.3 + log10(connectivity+1)*0.2
    # ★ boost +0.2 for conceptually central labels
    # ============================================================

    def get_node2vec_weights(self) -> Dict[str, float]:
        """UDL Node2Vec label weights based on actual 763-node graph"""
        return {
            # =====================================================
            # UDL FRAMEWORK (conceptually central — ★ boost)
            # =====================================================
            'Checkpoint': 1.95,                  # 55 nodes, high connectivity via MENTIONS ★
            'Guideline': 1.75,                   # 10 nodes, hub via MENTIONS/ALIGNS_TO ★
            'Principle': 1.55,                   # 2 nodes, hub via ALIGNS_TO ★
            'LearningFramework': 1.45,           # 2 nodes ★

            # =====================================================
            # TOOLS & RESOURCES (high node counts)
            # =====================================================
            'AnalogicalTool': 1.75,              # 36 nodes — target of SUPPORTS_BY
            'DigitalTool': 1.70,                 # 29 nodes — target of SUPPORTS_BY
            'UniversalTool': 1.45,               # 7 nodes
            'VisualTool': 1.35,                  # 4 nodes
            'UniversalStrategy': 1.45,           # 7 nodes

            # =====================================================
            # PEDAGOGICAL APPROACHES & STRATEGIES (★ boost)
            # =====================================================
            'EducationalApproach': 1.85,         # 26 nodes ★
            'InstructionalStrategy': 1.80,       # 21 nodes ★
            'InstructionalTechnique': 1.60,      # 11 nodes ★
            'LearningMethodology': 1.65,         # 10 nodes ★
            'PedagogicalSupports': 1.55,         # 8 nodes, hub via SUPPORTS_BY
            'LinguisticSupports': 1.50,          # 6 nodes, hub via SUPPORTS_BY

            # =====================================================
            # LEARNER VARIABILITY — DISABILITY / CHALLENGE TYPES (★)
            # =====================================================
            'Adhd': 1.65,                        # 10 nodes ★
            'AutismSpectrum': 1.65,              # 11 nodes ★
            'Dyscalculia': 1.65,                 # 12 nodes ★
            'Dyslexia': 1.55,                    # 7 nodes ★
            'Gifted': 1.60,                      # 10 nodes ★
            'ForeignStudents': 1.45,             # 5 nodes ★
            'SensoryDisabilities': 1.35,         # 3 nodes ★

            # =====================================================
            # BARRIERS
            # =====================================================
            'Barrier': 1.50,                     # 6 nodes — hub via ASSOCIATES_TO
            'SensoryBarrier': 1.55,              # 9 nodes
            'LinguisticBarrier': 1.50,           # 7 nodes
            'CognitiveBarrier': 1.40,            # 5 nodes
            'TechnologicalBarrier': 1.40,        # 5 nodes
            'ExecutiveBarrier': 1.30,            # 2 nodes

            # =====================================================
            # OBSERVABLE OUTCOMES & BEHAVIORS
            # =====================================================
            'BehavioralManifestations': 1.75,    # 26 nodes — target of LEADS
            'ObservableLearningOutcomes': 1.60,  # 13 nodes — source of MITIGATED_BY
            'LearningChallenge': 1.70,           # 20 nodes — target of PREDICTS, source of MITIGATED_BY ★
            'LearningOutcomes': 1.35,            # 4 nodes

            # =====================================================
            # COGNITIVE & EXECUTIVE FUNCTION
            # =====================================================
            'CognitiveProcesses': 1.40,          # 3 nodes — source of PREDICTS
            'AttentionalControlProcesses': 1.35, # 2 nodes
            'Metacognition': 1.35,               # 2 nodes
            'Attention': 1.30,                   # 2 nodes
            'CriticalThinking': 1.25,            # 1 node

            # =====================================================
            # MOTIVATIONAL & AFFECTIVE
            # =====================================================
            'PerceptionsOfTaskValue': 1.35,      # 2 nodes — source of LEADS
            'AttributionalStyle': 1.35,          # 2 nodes — source of LEADS
            'Mindset': 1.30,                     # 2 nodes
            'Safety': 1.30,                      # 2 nodes

            # =====================================================
            # CONTEXT & ENVIRONMENT
            # =====================================================
            'Context': 1.50,                     # 8 nodes — source of SUGGESTS
            'SensoryProcessing': 1.45,           # 7 nodes — target of ASSOCIATES_TO
            'SensoryEnvironment': 1.45,          # 7 nodes
            'SocioculturalPragmatic': 1.40,      # 6 nodes
            'Colour': 1.30,                      # 5 nodes
            'Furniture': 1.30,                   # 5 nodes
            'Lighting': 1.25,                    # 4 nodes
            'Acoustics': 1.25,                   # 3 nodes
            'Smells': 1.20,                      # 4 nodes
            'Textures': 1.20,                    # 4 nodes

            # =====================================================
            # ASSESSMENT
            # =====================================================
            'AssessmentDesign': 1.40,            # 5 nodes
            'TeachingAssessment': 1.35,          # 2 nodes — hub via INCLUDES/PROMOTES

            # =====================================================
            # LITERACY & LANGUAGE
            # =====================================================
            'LiteracyProcessing': 1.40,          # 6 nodes
            'ComprehensionProcessing': 1.35,     # 4 nodes
            'LanguageKnowledge': 1.30,           # 4 nodes
            'LanguageScaffolds': 1.30,           # 3 nodes

            # =====================================================
            # MITIGATION STRATEGIES (targets of MITIGATED_BY)
            # =====================================================
            'ExternalizedExecutiveSupport': 1.35,    # 2 nodes
            'InitiationCueing': 1.30,                # 2 nodes
            'RelevanceEnhancementStrategy': 1.30,    # 2 nodes
            'StrengthenedRecall': 1.30,              # 2 nodes
            'ConcreteBridgingStrategy': 1.30,        # 2 nodes
            'ConfidenceBuilding': 1.30,              # 2 nodes
            'EmpowermentStrategy': 1.30,             # 2 nodes
            'AccessibilityStrategy': 1.35,           # 4 nodes
            'UniversalDesignForDigitalAccess': 1.30, # 3 nodes
            'CognitiveFlexibilityTraining': 1.25,    # 2 nodes
            'ScaffoldedSkillDevelopment': 1.25,      # 2 nodes
            'AccommodationForProcessingSpeed': 1.25, # 2 nodes

            # =====================================================
            # DIGITAL LEARNING
            # =====================================================
            'DigitalLearning': 1.30,             # 4 nodes
            'Skills': 1.25,                      # 3 nodes
            'CreativeThinking': 1.25,            # 3 nodes
        }

    # ============================================================
    # VALID METHODOLOGY LABELS (for context_builder.py filtering)
    # ============================================================

    def get_valid_methodology_labels(self) -> List[str]:
        """All valid labels in the UDL graph (March 2026 — 271 unique).

        Only includes labels with >= 2 nodes or that are conceptually central.
        """
        return [
            # UDL Framework
            'Checkpoint', 'Guideline', 'Principle', 'LearningFramework', 'Framework',

            # Tools & Resources
            'AnalogicalTool', 'DigitalTool', 'UniversalTool', 'VisualTool',
            'TouchTool', 'Tool', 'UniversalStrategy',

            # Pedagogical Approaches & Strategies
            'EducationalApproach', 'InstructionalStrategy', 'InstructionalTechnique',
            'LearningMethodology', 'LearningStrategy', 'TeachingStrategy',
            'PedagogicalSupports', 'LinguisticSupports', 'CollaborativeStrategy',
            'CognitiveStrategy', 'MotivationalStrategy', 'MemoryStrategy',
            'FeedbackTechnique', 'AssessmentTechnique',

            # Learner Variability (disability types)
            'Adhd', 'AutismSpectrum', 'Dyscalculia', 'Dyslexia',
            'Gifted', 'ForeignStudents', 'SensoryDisabilities',
            'PhysicalDisabilities',

            # Barriers
            'Barrier', 'SensoryBarrier', 'LinguisticBarrier', 'CognitiveBarrier',
            'TechnologicalBarrier', 'ExecutiveBarrier', 'CollaborativeBarrier',

            # Observable Outcomes & Behaviors
            'BehavioralManifestations', 'ObservableLearningOutcomes',
            'ObservableBehavioralOutcomes', 'ObservableLiteracyOutcomes',
            'ObservableLearningChallenges', 'ObservableProblemSolvingOutcomes',
            'LearningChallenge', 'LearningOutcomes', 'BehavioralOutcome',
            'CognitiveOutcomes', 'UtilizationOutcomes', 'EmotionalOutcomes',
            'WorkflowOutcomes', 'IndividualOutcomes', 'UsabilityFailureOutcomes',

            # Cognitive & Executive Function
            'CognitiveProcesses', 'AttentionalControlProcesses', 'Metacognition',
            'Attention', 'CriticalThinking', 'PatternRecognition',
            'CognitiveIntegration', 'CognitiveLoad', 'CognitiveProcess',
            'MetacognitiveProcesses', 'SelfRegulationProcesses',
            'AdaptiveThinkingProcesses', 'ActionInitiationProcesses',
            'TemporalOrganizationProcesses', 'SustainedDirectiveProcesses',
            'CreativeThinking',

            # Motivational & Affective
            'PerceptionsOfTaskValue', 'AttributionalStyle', 'Mindset',
            'Safety', 'IntrinsicMotivation', 'GoalOrientation',
            'AffectiveState', 'PerformanceAnxiety', 'EffortBeliefs',
            'BeliefsAboutSelf', 'OutcomeExpectancy', 'RegulationOfBehavior',

            # Context & Environment
            'Context', 'SensoryProcessing', 'SensoryEnvironment',
            'SocioculturalPragmatic', 'Colour', 'Furniture', 'Lighting',
            'Acoustics', 'Smells', 'Textures', 'PhysicalEnvironment',
            'InteractiveBoard', 'ClassroomClimate', 'EnvironmentalStressor',

            # Assessment
            'AssessmentDesign', 'TeachingAssessment',

            # Literacy & Language
            'LiteracyProcessing', 'ComprehensionProcessing', 'LanguageKnowledge',
            'LanguageScaffolds', 'VisualResources',

            # Mitigation & Support Strategies
            'ExternalizedExecutiveSupport', 'InitiationCueing',
            'RelevanceEnhancementStrategy', 'StrengthenedRecall',
            'ConcreteBridgingStrategy', 'ConfidenceBuilding',
            'EmpowermentStrategy', 'AccessibilityStrategy',
            'UniversalDesignForDigitalAccess', 'CognitiveFlexibilityTraining',
            'ScaffoldedSkillDevelopment', 'AccommodationForProcessingSpeed',
            'MnemonicsTechniques', 'AttentionStaminaTraining',

            # Sensory & Environmental
            'SensoryOverload', 'SensoryAccommodation',
            'InclusiveSensoryDesign', 'SensoryRegulation',
            'SensoryBufferingStrategy', 'ControlledInputStrategy',
            'PhysicalWellBeing', 'PhysiologicalDistraction',

            # Digital
            'DigitalLearning', 'Skills', 'InclusiveLearning',

            # Assessment Bloom-style (from Objective → Suggests)
            'Objective', 'Recall', 'Summarize', 'Use', 'Compare', 'Judge', 'Design',
            'UnderstandingProcess', 'KnowledgeApplicationProcess',
            'ActiveCognitiveProcess', 'EvaluateProcess', 'OutcomeProcess',

            # Fallback
            'Concept',
        ]

    # ============================================================
    # LABEL → CATEGORY MAPPING (for context_builder.py)
    # ============================================================

    def get_label_category_map(self) -> Dict[str, str]:
        """Map Neo4j node labels to human-readable Italian category names."""
        return {
            # UDL Framework
            'Checkpoint': 'Checkpoint UDL',
            'Guideline': 'Linea Guida UDL',
            'Principle': 'Principio UDL',
            'LearningFramework': 'Framework Educativo',
            'Framework': 'Framework Educativo',

            # Tools
            'AnalogicalTool': 'Strumento Analogico',
            'DigitalTool': 'Strumento Digitale',
            'UniversalTool': 'Strumento Universale',
            'VisualTool': 'Strumento Visivo',
            'TouchTool': 'Strumento Tattile',
            'Tool': 'Strumento',
            'InteractiveBoard': 'Lavagna Interattiva',

            # Pedagogical Approaches
            'EducationalApproach': 'Approccio Educativo',
            'InstructionalStrategy': 'Strategia Didattica',
            'InstructionalTechnique': 'Tecnica Didattica',
            'LearningMethodology': 'Metodologia di Apprendimento',
            'LearningStrategy': 'Strategia di Apprendimento',
            'TeachingStrategy': 'Strategia di Insegnamento',
            'PedagogicalSupports': 'Supporto Pedagogico',
            'LinguisticSupports': 'Supporto Linguistico',
            'CollaborativeStrategy': 'Strategia Collaborativa',
            'CognitiveStrategy': 'Strategia Cognitiva',
            'MotivationalStrategy': 'Strategia Motivazionale',
            'MemoryStrategy': 'Strategia di Memoria',
            'FeedbackTechnique': 'Tecnica di Feedback',
            'UniversalStrategy': 'Strategia Universale',
            'UniversalSupport': 'Supporto Universale',

            # Learner Variability
            'Adhd': 'Variabilità – ADHD',
            'AutismSpectrum': 'Variabilità – Spettro Autistico',
            'Dyscalculia': 'Variabilità – Discalculia',
            'Dyslexia': 'Variabilità – Dislessia',
            'Gifted': 'Variabilità – Plusdotazione',
            'ForeignStudents': 'Variabilità – Studenti Stranieri',
            'SensoryDisabilities': 'Variabilità – Disabilità Sensoriali',
            'PhysicalDisabilities': 'Variabilità – Disabilità Fisiche',

            # Barriers
            'Barrier': 'Barriera',
            'SensoryBarrier': 'Barriera Sensoriale',
            'LinguisticBarrier': 'Barriera Linguistica',
            'CognitiveBarrier': 'Barriera Cognitiva',
            'TechnologicalBarrier': 'Barriera Tecnologica',
            'ExecutiveBarrier': 'Barriera Esecutiva',
            'CollaborativeBarrier': 'Barriera Collaborativa',

            # Observable Outcomes
            'BehavioralManifestations': 'Manifestazione Comportamentale',
            'ObservableLearningOutcomes': 'Esito Osservabile',
            'ObservableBehavioralOutcomes': 'Esito Comportamentale Osservabile',
            'ObservableLiteracyOutcomes': 'Esito Osservabile – Literacy',
            'ObservableLearningChallenges': 'Sfida Osservabile',
            'LearningChallenge': 'Sfida di Apprendimento',
            'LearningOutcomes': 'Esito di Apprendimento',
            'BehavioralOutcome': 'Esito Comportamentale',
            'CognitiveOutcomes': 'Esito Cognitivo',
            'UtilizationOutcomes': 'Esito d\'Utilizzo',
            'EmotionalOutcomes': 'Esito Emotivo',

            # Cognitive & Executive
            'CognitiveProcesses': 'Processo Cognitivo',
            'CognitiveProcess': 'Processo Cognitivo',
            'AttentionalControlProcesses': 'Controllo Attentivo',
            'Metacognition': 'Metacognizione',
            'MetacognitiveProcesses': 'Processi Metacognitivi',
            'Attention': 'Attenzione',
            'CriticalThinking': 'Pensiero Critico',
            'CreativeThinking': 'Pensiero Creativo',
            'PatternRecognition': 'Riconoscimento di Pattern',
            'CognitiveIntegration': 'Integrazione Cognitiva',
            'CognitiveLoad': 'Carico Cognitivo',
            'CognitiveFlexibilityTraining': 'Training Flessibilità Cognitiva',
            'SelfRegulationProcesses': 'Processi di Autoregolazione',

            # Motivational & Affective
            'PerceptionsOfTaskValue': 'Percezione del Valore del Compito',
            'AttributionalStyle': 'Stile Attributivo',
            'Mindset': 'Mentalità',
            'Safety': 'Sicurezza',
            'IntrinsicMotivation': 'Motivazione Intrinseca',
            'GoalOrientation': 'Orientamento agli Obiettivi',
            'AffectiveState': 'Stato Affettivo',
            'PerformanceAnxiety': 'Ansia da Prestazione',
            'ConfidenceBuilding': 'Costruzione della Fiducia',

            # Context & Environment
            'Context': 'Contesto Classe',
            'SensoryProcessing': 'Elaborazione Sensoriale',
            'SensoryEnvironment': 'Ambiente Sensoriale',
            'SocioculturalPragmatic': 'Aspetto Socioculturale',
            'Colour': 'Colore (Ambiente)',
            'Furniture': 'Arredamento',
            'Lighting': 'Illuminazione',
            'Acoustics': 'Acustica',
            'Smells': 'Odori',
            'Textures': 'Texture',
            'PhysicalEnvironment': 'Ambiente Fisico',
            'ClassroomClimate': 'Clima di Classe',

            # Assessment
            'AssessmentDesign': 'Progettazione Valutativa',
            'TeachingAssessment': 'Valutazione Didattica',

            # Literacy & Language
            'LiteracyProcessing': 'Elaborazione della Lettura',
            'ComprehensionProcessing': 'Elaborazione della Comprensione',
            'LanguageKnowledge': 'Conoscenza Linguistica',
            'LanguageScaffolds': 'Scaffolding Linguistico',

            # Mitigation
            'ExternalizedExecutiveSupport': 'Supporto Esecutivo Esternalizzato',
            'InitiationCueing': 'Cueing per l\'Iniziazione',
            'RelevanceEnhancementStrategy': 'Strategia di Rilevanza',
            'StrengthenedRecall': 'Rinforzo della Memoria',
            'ConcreteBridgingStrategy': 'Strategia di Bridging Concreto',
            'EmpowermentStrategy': 'Strategia di Empowerment',
            'AccessibilityStrategy': 'Strategia di Accessibilità',
            'AccommodationForProcessingSpeed': 'Adattamento per Velocità',
            'ScaffoldedSkillDevelopment': 'Sviluppo Competenze Scaffoldato',
            'MnemonicsTechniques': 'Tecniche Mnemoniche',

            # Digital
            'DigitalLearning': 'Apprendimento Digitale',
            'InclusiveLearning': 'Apprendimento Inclusivo',

            # Bloom-style Assessment
            'Objective': 'Obiettivo Didattico',
            'Recall': 'Ricordare',
            'Summarize': 'Riassumere',
            'Use': 'Applicare',
            'Compare': 'Confrontare',
            'Judge': 'Valutare',
            'Design': 'Progettare',

            # Fallback
            'Concept': 'Concetto Educativo',
        }

    # ============================================================
    # RETRIEVAL BOOSTS
    # ============================================================

    def get_retrieval_boosts(self) -> Dict[str, Dict[str, float]]:
        """UDL retrieval boosts based on 763-node graph (March 2026)"""
        return {
            'label_boosts': {
                # UDL Framework (conceptually central)
                'Checkpoint': 2.0,
                'Guideline': 1.9,
                'Principle': 1.8,

                # Pedagogical core
                'EducationalApproach': 2.0,
                'InstructionalStrategy': 2.0,
                'LearningMethodology': 1.9,
                'InstructionalTechnique': 1.7,
                'PedagogicalSupports': 1.7,
                'LinguisticSupports': 1.6,

                # Tools
                'AnalogicalTool': 1.7,
                'DigitalTool': 1.7,
                'UniversalTool': 1.5,
                'VisualTool': 1.5,

                # Learner variability (★ high priority)
                'Adhd': 2.0,
                'AutismSpectrum': 2.0,
                'Dyscalculia': 2.0,
                'Dyslexia': 2.0,
                'Gifted': 1.8,
                'ForeignStudents': 1.8,
                'SensoryDisabilities': 1.7,

                # Barriers
                'Barrier': 1.7,
                'SensoryBarrier': 1.6,
                'LinguisticBarrier': 1.6,
                'CognitiveBarrier': 1.5,
                'TechnologicalBarrier': 1.5,

                # Outcomes & challenges
                'BehavioralManifestations': 1.6,
                'LearningChallenge': 1.7,
                'ObservableLearningOutcomes': 1.5,
                'LearningOutcomes': 1.5,

                # Cognitive & Executive
                'CognitiveProcesses': 1.5,
                'AttentionalControlProcesses': 1.4,
                'Metacognition': 1.5,

                # Context & Environment
                'Context': 1.6,
                'SensoryProcessing': 1.4,
                'SensoryEnvironment': 1.4,

                # Mitigation strategies
                'ExternalizedExecutiveSupport': 1.4,
                'AccessibilityStrategy': 1.5,
                'UniversalDesignForDigitalAccess': 1.4,

                # Assessment
                'AssessmentDesign': 1.5,
                'TeachingAssessment': 1.5,
            },
            'relationship_boosts': {
                'MITIGATED_BY': 1.5,
                'SUGGESTS': 1.4,
                'ASSOCIATES_TO': 1.3,
                'SUPPORTS_BY': 1.3,
                'MENTIONS': 1.2,
                'LEADS': 1.3,
                'CAUSES': 1.3,
                'NO_SUGGESTS': 1.3,
                'PREDICTS': 1.3,
                'PROMOTES': 1.2,
                'SUPPORTS': 1.3,
                'FOSTERS': 1.2,
                'INCREASES': 1.2,
                'ALIGNS_TO': 1.2,
                'INCLUDES': 1.2,
                'REDUCES': 1.2,
                'FACILITATES': 1.2,
            }
        }

    def get_similarity_threshold(self) -> float:
        """UDL similarity threshold — moderate, graph has high label diversity"""
        return 0.72

    # ============================================================
    # TEXT2CYPHER FEW-SHOT EXAMPLES
    # Based on actual graph patterns (March 2026)
    # ============================================================

    def get_few_shot_examples(self, domain: str = "udl") -> str:
        """UDL-specific few-shot examples for Cypher generation.

        Built from the 30 most frequent relationship patterns in the
        763-node UDL graph (March 2026 ingestion).
        """
        examples = """
Question: "What strategies help students with ADHD?"
Cypher: MATCH (a:Adhd {{domain: "{domain}"}})-[r:SUGGESTS]->(m {{domain: "{domain}"}}) RETURN a.name AS challenge, type(r) AS relationship, m.name AS strategy, labels(m) AS strategy_type LIMIT 15

Question: "What approaches should be avoided for students with dyscalculia?"
Cypher: MATCH (d:Dyscalculia {{domain: "{domain}"}})-[r:NO_SUGGESTS]->(m {{domain: "{domain}"}}) RETURN d.name AS challenge, m.name AS approach_to_avoid, labels(m) AS type LIMIT 10

Question: "What are the UDL checkpoints?"
Cypher: MATCH (g:Guideline {{domain: "{domain}"}})-[r:MENTIONS]->(c:Checkpoint {{domain: "{domain}"}}) RETURN g.name AS guideline, c.name AS checkpoint LIMIT 20

Question: "What tools support pedagogical approaches?"
Cypher: MATCH (p:PedagogicalSupports {{domain: "{domain}"}})-[r:SUPPORTS_BY]->(t {{domain: "{domain}"}}) RETURN p.name AS support, t.name AS tool, labels(t) AS tool_type LIMIT 15

Question: "What barriers are associated with sensory processing?"
Cypher: MATCH (b:Barrier {{domain: "{domain}"}})-[r:ASSOCIATES_TO]->(s:SensoryProcessing {{domain: "{domain}"}}) RETURN b.name AS barrier, s.name AS sensory_issue LIMIT 10

Question: "How can learning challenges in executive functions be mitigated?"
Cypher: MATCH (lc:LearningChallenge {{domain: "{domain}"}})-[r:MITIGATED_BY]->(s {{domain: "{domain}"}}) RETURN lc.name AS challenge, s.name AS strategy, labels(s) AS strategy_type LIMIT 15

Question: "What methodologies work for different class contexts?"
Cypher: MATCH (c:Context {{domain: "{domain}"}})-[r:SUGGESTS]->(m:LearningMethodology {{domain: "{domain}"}}) RETURN c.name AS context, m.name AS methodology LIMIT 10

Question: "What strategies help students on the autism spectrum?"
Cypher: MATCH (a:AutismSpectrum {{domain: "{domain}"}})-[r:SUGGESTS]->(m {{domain: "{domain}"}}) RETURN a.name AS characteristic, type(r) AS relationship, m.name AS strategy, labels(m) AS type LIMIT 15

Question: "What linguistic tools are available for foreign students?"
Cypher: MATCH (ls:LinguisticSupports {{domain: "{domain}"}})-[r:SUPPORTS_BY]->(t {{domain: "{domain}"}}) RETURN ls.name AS support, t.name AS tool, labels(t) AS tool_type LIMIT 15

Question: "What behavioral manifestations are caused by motivational factors?"
Cypher: MATCH (p:PerceptionsOfTaskValue {{domain: "{domain}"}})-[r:LEADS]->(b:BehavioralManifestations {{domain: "{domain}"}}) RETURN p.name AS factor, b.name AS manifestation LIMIT 10

Question: "How to mitigate observable learning outcomes?"
Cypher: MATCH (o:ObservableLearningOutcomes {{domain: "{domain}"}})-[r:MITIGATED_BY]->(s {{domain: "{domain}"}}) RETURN o.name AS outcome, s.name AS strategy, labels(s) AS type LIMIT 15

Question: "What strategies help students with dyslexia?"
Cypher: MATCH (d:Dyslexia {{domain: "{domain}"}})-[r:SUGGESTS]->(m {{domain: "{domain}"}}) RETURN d.name AS characteristic, m.name AS strategy, labels(m) AS type LIMIT 10

Question: "What approaches work for gifted students?"
Cypher: MATCH (g:Gifted {{domain: "{domain}"}})-[r:SUGGESTS]->(m {{domain: "{domain}"}}) RETURN g.name AS characteristic, m.name AS strategy, labels(m) AS type LIMIT 10

Question: "What are the UDL principles and their guidelines?"
Cypher: MATCH (p:Principle {{domain: "{domain}"}})-[r:ALIGNS_TO]->(g:Guideline {{domain: "{domain}"}}) RETURN p.name AS principle, g.name AS guideline LIMIT 10

Question: "What cognitive processes predict learning challenges?"
Cypher: MATCH (cp:CognitiveProcesses {{domain: "{domain}"}})-[r:PREDICTS]->(lc:LearningChallenge {{domain: "{domain}"}}) RETURN cp.name AS process, lc.name AS challenge LIMIT 10

Question: "What types of assessment design exist?"
Cypher: MATCH (ta:TeachingAssessment {{domain: "{domain}"}})-[r:INCLUDES]->(ad:AssessmentDesign {{domain: "{domain}"}}) RETURN ta.name AS assessment, ad.name AS design LIMIT 10

Question: "How many nodes are in the UDL knowledge graph?"
Cypher: MATCH (n {{domain: "{domain}"}}) RETURN count(n) AS total_nodes

Question: "What sensory barriers exist and how do they affect learning?"
Cypher: MATCH (sb:SensoryBarrier {{domain: "{domain}"}})-[r]->(target {{domain: "{domain}"}}) RETURN sb.name AS barrier, type(r) AS relationship, target.name AS effect, labels(target) AS type LIMIT 15

Question: "Come posso supportare studenti con ADHD?"
Cypher: MATCH (a:Adhd {{domain: "{domain}"}})-[r:SUGGESTS]->(m {{domain: "{domain}"}}) RETURN a.name AS challenge, m.name AS strategy, labels(m) AS type LIMIT 15

Question: "Quali strumenti digitali supportano l'apprendimento?"
Cypher: MATCH (p {{domain: "{domain}"}})-[r:SUPPORTS_BY]->(d:DigitalTool {{domain: "{domain}"}}) RETURN p.name AS support, d.name AS tool LIMIT 15

Question: "Quali approcci evitare per studenti dislessici?"
Cypher: MATCH (d:Dyslexia {{domain: "{domain}"}})-[r:NO_SUGGESTS]->(m {{domain: "{domain}"}}) RETURN d.name AS characteristic, m.name AS approach_to_avoid, labels(m) AS type LIMIT 10
"""
        return examples.replace("{domain}", domain).strip()

    def get_cypher_patterns(self) -> str:
        """UDL-specific Cypher patterns for system prompt"""
        return """
QUERY PATTERNS (UDL — Universal Design for Learning):

1. Learner variability → Strategies (what helps):
   MATCH (learner:Adhd|AutismSpectrum|Dyscalculia|Dyslexia|Gifted|ForeignStudents)-[r:SUGGESTS]->(strategy)
   WHERE learner.domain = "udl"

2. Learner variability → Approaches to avoid:
   MATCH (learner:Dyscalculia|Dyslexia)-[r:NO_SUGGESTS]->(approach)

3. UDL Framework (Principles → Guidelines → Checkpoints):
   MATCH (p:Principle)-[:ALIGNS_TO]->(g:Guideline)-[:MENTIONS]->(c:Checkpoint)

4. Support → Tools (digital/analogical):
   MATCH (support:PedagogicalSupports|LinguisticSupports)-[:SUPPORTS_BY]->(tool:DigitalTool|AnalogicalTool)

5. Barriers → Sensory/cognitive/linguistic:
   MATCH (b:Barrier)-[:ASSOCIATES_TO]->(target:SensoryProcessing|SocioculturalPragmatic|SensoryEnvironment)

6. Learning challenges → Mitigation strategies:
   MATCH (lc:LearningChallenge)-[:MITIGATED_BY]->(s)

7. Cognitive processes → Learning challenges (prediction):
   MATCH (cp:CognitiveProcesses|AttentionalControlProcesses)-[:PREDICTS]->(lc:LearningChallenge)

8. Motivational factors → Behavioral manifestations:
   MATCH (factor:PerceptionsOfTaskValue|AttributionalStyle)-[:LEADS]->(bm:BehavioralManifestations)

9. Context → Methodologies:
   MATCH (c:Context)-[:SUGGESTS]->(m:LearningMethodology)

10. Assessment design:
    MATCH (ta:TeachingAssessment)-[:INCLUDES]->(ad:AssessmentDesign)

11. Environmental factors:
    MATCH (env:Lighting|Colour|Furniture|Acoustics|Smells|Textures)-[r]->(outcome)

CRITICAL RULES:
- Use EXACT node labels (case-sensitive!): Adhd, AutismSpectrum, Dyscalculia, Dyslexia, Gifted, ForeignStudents
- Top relationship types: MITIGATED_BY, SUGGESTS, ASSOCIATES_TO, SUPPORTS_BY, MENTIONS, LEADS, CAUSES, NO_SUGGESTS, PREDICTS
- Always add domain filter: {{domain: "udl"}}
- Node names are in English (e.g., "Difficulty sustaining focus", "Cooperative Learning")
"""

    def repair_cypher_query(self, query: str) -> str:
        """UDL-specific Cypher repair logic for the new 763-node graph"""

        # 1) Fix OLD labels from the previous UDL graph → NEW labels
        old_to_new = {
            'StudentWithSpecialNeeds': None,
            'PedagogicalMethodology': 'EducationalApproach',
            'PedagogicalStrategy': 'InstructionalStrategy',
            'StudentCharacteristic': None,
            'LearningResource': 'DigitalTool',
            'EnvironmentalBarrier': 'Barrier',
            'EnvironmentalSupport': 'PedagogicalSupports',
        }
        for old_label, new_label in old_to_new.items():
            if old_label.lower() in query.lower():
                if new_label:
                    query = re.sub(rf'\b{old_label}\b', new_label, query, flags=re.IGNORECASE)

        # 2) Fix case-sensitivity for learner variability labels
        case_fixes = {
            'adhd': 'Adhd',
            'autismspectrum': 'AutismSpectrum',
            'dyscalculia': 'Dyscalculia',
            'dyslexia': 'Dyslexia',
            'gifted': 'Gifted',
            'foreignstudents': 'ForeignStudents',
            'sensorydisabilities': 'SensoryDisabilities',
        }
        for lowercase, correct in case_fixes.items():
            if lowercase in query.lower() and correct not in query:
                query = re.sub(rf'\b{lowercase}\b', correct, query, flags=re.IGNORECASE)

        # 3) Fix case-sensitivity for pedagogical labels
        peda_fixes = {
            'educationalapproach': 'EducationalApproach',
            'instructionalstrategy': 'InstructionalStrategy',
            'instructionaltechnique': 'InstructionalTechnique',
            'learningmethodology': 'LearningMethodology',
            'pedagogicalsupports': 'PedagogicalSupports',
            'linguisticsupports': 'LinguisticSupports',
            'learningchallenge': 'LearningChallenge',
            'behavioralmanifestations': 'BehavioralManifestations',
            'observablelearningoutcomes': 'ObservableLearningOutcomes',
        }
        for lowercase, correct in peda_fixes.items():
            if lowercase in query.lower() and correct not in query:
                query = re.sub(rf'\b{lowercase}\b', correct, query, flags=re.IGNORECASE)

        # 4) Fix UDL framework labels
        framework_fixes = {
            'checkpoint': 'Checkpoint',
            'guideline': 'Guideline',
            'principle': 'Principle',
        }
        for lowercase, correct in framework_fixes.items():
            pattern = rf':({lowercase})\b'
            if re.search(pattern, query, re.IGNORECASE):
                query = re.sub(pattern, f':{correct}', query, flags=re.IGNORECASE)

        # 5) Fix tool labels
        tool_fixes = {
            'analogicaltool': 'AnalogicalTool',
            'digitaltool': 'DigitalTool',
            'universaltool': 'UniversalTool',
            'visualtool': 'VisualTool',
            'interactiveboard': 'InteractiveBoard',
        }
        for lowercase, correct in tool_fixes.items():
            if lowercase in query.lower() and correct not in query:
                query = re.sub(rf'\b{lowercase}\b', correct, query, flags=re.IGNORECASE)

        # 6) Fix UNION column mismatches (same logic as Neuro)
        if 'UNION' in query:
            query = self._repair_union_columns(query)

        return query

    def _repair_union_columns(self, query: str) -> str:
        """Fix UNION queries with mismatched column names."""
        parts = query.split('UNION')
        if len(parts) < 2:
            return query

        fixed_parts = []
        for part in parts:
            part = part.strip()
            match_pattern = re.search(r'MATCH\s+\((\w+):', part)
            if not match_pattern:
                fixed_parts.append(part)
                continue

            var_name = match_pattern.group(1)

            simple_pattern = rf'RETURN\s+{var_name}\s*,\s*labels\({var_name}\)\s+as\s+node_labels'
            if re.search(simple_pattern, part, re.IGNORECASE):
                fixed_part = re.sub(
                    simple_pattern,
                    f'RETURN {var_name} AS concept, labels({var_name}) as node_labels',
                    part, flags=re.IGNORECASE
                )
                fixed_parts.append(fixed_part)
                continue

            type_pattern = rf'RETURN\s+"[^"]+"\s+as\s+type\s*,\s*{var_name}\s*,\s*labels\({var_name}\)\s+as\s+node_labels'
            if re.search(type_pattern, part, re.IGNORECASE):
                fixed_part = re.sub(
                    rf'RETURN\s+("[^"]+")\s+as\s+type\s*,\s*{var_name}\s*,\s*labels\({var_name}\)\s+as\s+node_labels',
                    rf'RETURN \1 as type, {var_name} AS concept, labels({var_name}) as node_labels',
                    part, flags=re.IGNORECASE
                )
                fixed_parts.append(fixed_part)
                continue

            fixed_parts.append(part)

        result = ' UNION '.join(fixed_parts)
        if result != query:
            import logging
            logging.getLogger(__name__).info(
                f"[UNION Fix] Standardized column names in {len(parts)}-way UNION query"
            )
        return result

    # ============================================================
    # MULTILINGUAL TERMS (Italian → English)
    # ============================================================

    def get_italian_terms(self) -> Dict[str, str]:
        """UDL Italian→English term mapping based on actual node names"""
        return {
            # Learner Variability — mapped to actual label names
            "ADHD": "Adhd",
            "deficit di attenzione": "Adhd",
            "iperattività": "Adhd",
            "disturbo dell'attenzione": "Adhd",

            "autismo": "AutismSpectrum",
            "spettro autistico": "AutismSpectrum",
            "disturbi dello spettro autistico": "AutismSpectrum",

            "discalculia": "Dyscalculia",
            "difficoltà in matematica": "Dyscalculia",

            "dislessia": "Dyslexia",
            "difficoltà di lettura": "Dyslexia",

            "plusdotazione": "Gifted",
            "studenti plusdotati": "Gifted",
            "eccellenza": "Gifted",
            "alto potenziale": "Gifted",

            "studenti stranieri": "ForeignStudents",
            "alunni stranieri": "ForeignStudents",
            "difficoltà linguistiche": "ForeignStudents",
            "barriere linguistiche": "ForeignStudents",
            "L2": "ForeignStudents",

            "disabilità sensoriali": "SensoryDisabilities",
            "ipovedenti": "SensoryDisabilities",
            "non vedenti": "SensoryDisabilities",
            "sordi": "SensoryDisabilities",

            # Barriers
            "barriere": "Barrier",
            "barriera sensoriale": "SensoryBarrier",
            "barriera linguistica": "LinguisticBarrier",
            "barriera cognitiva": "CognitiveBarrier",
            "barriera tecnologica": "TechnologicalBarrier",

            # Teaching Approaches — mapped to actual node names
            "apprendimento cooperativo": "Cooperative Learning",
            "flipped classroom": "Flipped Classroom",
            "classe capovolta": "Flipped Classroom",
            "gamification": "Gammification",
            "ludicizzazione": "Gammification",
            "project based learning": "Project Based Learning",
            "apprendimento basato su progetti": "Project Based Learning",
            "peer to peer": "Peer to Peer",
            "tutoraggio tra pari": "Peer to Peer",
            "scaffolding": "Scaffolding",
            "istruzione differenziata": "Differentiated Instruction",
            "lezione frontale": "Long Frontal Lesson",
            "apprendimento passivo": "Passive Learning",
            "autoregolazione": "Self-Regulated Learning",
            "challenge based learning": "Challenge-Based Learning",

            # UDL Framework
            "UDL": "Universal Design for Learning",
            "progettazione universale": "Universal Design for Learning",
            "principi UDL": "UDL principles",
            "linee guida UDL": "UDL guidelines",
            "checkpoint UDL": "UDL checkpoints",
            "framework UDL": "UDL framework",

            # Tools
            "strumenti digitali": "DigitalTool",
            "strumenti analogici": "AnalogicalTool",
            "strumenti visivi": "VisualTool",
            "lavagna interattiva": "InteractiveBoard",

            # Environment
            "illuminazione": "Lighting",
            "acustica": "Acoustics",
            "arredamento": "Furniture",
            "colori": "Colour",
            "odori": "Smells",
            "texture": "Textures",

            # Cognitive / Executive
            "funzioni esecutive": "executive functions",
            "sfide di apprendimento": "learning challenges",
            "processi cognitivi": "cognitive processes",
            "metacognizione": "Metacognition",
            "controllo attentivo": "attentional control",
            "pensiero critico": "critical thinking",
            "carico cognitivo": "cognitive load",

            # Motivational
            "motivazione": "motivation",
            "motivazione intrinseca": "intrinsic motivation",
            "ansia da prestazione": "performance anxiety",
            "mentalità": "mindset",

            # Assessment
            "valutazione": "assessment",
            "valutazione autentica": "Authentic Task",
            "rubrica": "assessment design",

            # Context
            "classe eterogenea": "heterogeneous class",
            "contesto educativo": "educational context",
            "ambiente inclusivo": "inclusive environment",
            "ambiente di apprendimento": "learning environment",

            # Generic pedagogical
            "strategie": "strategies",
            "metodologie": "methodologies",
            "inclusione": "inclusion",
            "accessibilità": "accessibility",
            "differenziazione": "differentiation",
            "personalizzazione": "personalization",
            "supporto visivo": "visual supports",
            "multisensoriale": "multisensory",
            "attività multisensoriali": "Multisensory Activities",
        }

    def get_query_context(self) -> str:
        return "education, inclusive pedagogy, and Universal Design for Learning"

    # ============================================================
    # LLM SYSTEM PROMPT
    # ============================================================

    def get_system_prompt(self) -> str:
        """Fetch the UDL system prompt from Langfuse (legacy GraphRAG mode)."""
        from aix.domains.langfuse_prompts import fetch_prompt
        return fetch_prompt("udl.system_prompt")

    def get_writer_prompt(self) -> str:
        """Fetch the UDL writer prompt from Langfuse (agent mode)."""
        from aix.domains.langfuse_prompts import fetch_prompt
        return fetch_prompt("udl.writer_prompt")

    def get_response_template(self) -> str:
        """Fetch the UDL response template from Langfuse (legacy GraphRAG mode)."""
        from aix.domains.langfuse_prompts import fetch_prompt
        return fetch_prompt("udl.response_template")

    # ============================================================
    # CONTEXT BUILDER — METHODOLOGY CATEGORIES
    # ============================================================

    def get_methodology_categories(self) -> Dict:
        """UDL methodology categories based on actual graph nodes"""
        return {
            'Cooperative Learning': {
                'category': 'Collaborative Pedagogy',
                'best_for': ['social_interaction', 'peer_learning', 'inclusion', 'foreign_students'],
                'implementation': 'Organize students in diverse groups of 3-5 members',
                'applications': [
                    'Jigsaw method for complex topics',
                    'Think-Pair-Share for quick engagement',
                    'Peer tutoring for language support'
                ],
                'special_needs_adaptations': [
                    'Assign complementary roles based on abilities',
                    'Provide visual and verbal instructions',
                    'Use translanguaging for multilingual students'
                ]
            },
            'Flipped Classroom': {
                'category': 'Blended Learning',
                'best_for': ['self_paced_learning', 'differentiation', 'gifted_students'],
                'implementation': 'Pre-recorded content at home, active learning in class',
                'applications': [
                    'Video lectures with closed captions',
                    'Interactive activities during class time',
                    'Personalized learning paths'
                ],
                'special_needs_adaptations': [
                    'Multiple representation formats (audio + visual + text)',
                    'Flexible pacing for processing speed differences',
                    'Subtitles and text-to-speech options'
                ]
            },
            'Project Based Learning': {
                'category': 'Constructivist Pedagogy',
                'best_for': ['deep_learning', 'critical_thinking', 'gifted_engagement'],
                'implementation': 'Students work on extended projects addressing real problems',
                'applications': [
                    'Research projects with presentation',
                    'Design challenges with multiple expression modes',
                    'Community service projects'
                ],
                'special_needs_adaptations': [
                    'Scaffold project steps with visual checklists',
                    'Provide multiple representation and expression options',
                    'Allow flexible demonstration methods (UDL Principle C)'
                ]
            },
            'Gammification': {
                'category': 'Engagement Strategy',
                'best_for': ['motivation', 'adhd_engagement', 'reward_sensitivity'],
                'implementation': 'Use game-based elements to boost engagement',
                'applications': [
                    'Digital educational games',
                    'Gamified math activities for dyscalculia',
                    'Challenge-based learning paths'
                ],
                'special_needs_adaptations': [
                    'Adjust difficulty levels dynamically',
                    'Provide multimodal interfaces',
                    'Use intrinsic reward structures'
                ]
            },
            'Scaffolding': {
                'category': 'Instructional Support',
                'best_for': ['executive_functions', 'processing_challenges', 'all_learners'],
                'implementation': 'Graduated support that fades as competence grows',
                'applications': [
                    'Goal-Plan-Do-Review framework',
                    'Visual organizers and checklists',
                    'Linguistic scaffolds for L2 learners'
                ],
                'special_needs_adaptations': [
                    'Externalized executive support (visual timers, planners)',
                    'Initiation cueing for task start',
                    'Accommodation for processing speed'
                ]
            },
            'Differentiated Instruction': {
                'category': 'Inclusive Pedagogy',
                'best_for': ['heterogeneous_classes', 'multiple_variabilities', 'UDL_alignment'],
                'implementation': 'Adapt content, process, product, and environment to learner needs',
                'applications': [
                    'Tiered activities by complexity',
                    'Learning stations with varied modalities',
                    'Choice boards for expression'
                ],
                'special_needs_adaptations': [
                    'Multiple means of engagement (UDL Principle A)',
                    'Multiple means of representation (UDL Principle B)',
                    'Multiple means of action and expression (UDL Principle C)'
                ]
            },
            'Multisensory Activities': {
                'category': 'Multi-Modal Instruction',
                'best_for': ['sensory_processing', 'dyslexia', 'engagement'],
                'implementation': 'Engage multiple senses in learning activities',
                'applications': [
                    'Visual + auditory + kinesthetic instruction',
                    'Hands-on manipulatives for math concepts',
                    'Movement-integrated learning'
                ],
                'special_needs_adaptations': [
                    'Adapt for sensory sensitivities (autism spectrum)',
                    'Provide sensory breaks and regulation tools',
                    'Use controlled input strategies'
                ]
            },
            'Self-Regulated Learning': {
                'category': 'Metacognitive Development',
                'best_for': ['executive_functions', 'metacognition', 'gifted_autonomy'],
                'implementation': 'Teach students to plan, monitor, and evaluate their own learning',
                'applications': [
                    'Self-questioning strategies',
                    'Learning journals and reflection',
                    'Goal setting with visible progress tracking'
                ],
                'special_needs_adaptations': [
                    'External memory aids for ADHD',
                    'Metacognitive guides for autism spectrum',
                    'Temporal organization support tools'
                ]
            },
        }

    # ============================================================
    # SPECIAL NEEDS MAPPING
    # ============================================================

    def get_special_needs_mapping(self) -> Dict:
        """UDL special needs mapping based on actual graph data"""
        return {
            'Adhd': {
                'primary_characteristics': [
                    'Difficulty sustaining focus',
                    'Impaired inhibitory control',
                    'Reduced working memory capacity',
                    'Altered reward sensitivity',
                    'Difficulty managing frustration or stress',
                    'Limited metacognitive monitoring'
                ],
                'recommended_methodologies': [
                    'Gammification', 'Cooperative Learning', 'Multisensory Activities',
                    'Focus Tasks', 'Scaffolding'
                ],
                'support_needs': ['externalized_executive_support', 'initiation_cueing',
                                  'focus_tasks', 'movement_breaks'],
                'environmental_factors': ['reduced_distractions', 'visual_schedules',
                                          'predictable_routines']
            },
            'AutismSpectrum': {
                'primary_characteristics': [
                    'Difficulty with theory of mind',
                    'Pragmatic language challenges',
                    'Hyper- or hypo-sensitivity',
                    'Rigidity, reduced cognitive flexibility',
                    'Strong focus on restricted interests',
                    'Preference for intrinsic, interest-driven tasks'
                ],
                'recommended_methodologies': [
                    'Predictable routines', 'Visual Tools', 'Explicit Social Skills Instruction',
                    'Self-Directed Exploration', 'Scaffolding'
                ],
                'support_needs': ['visual_structure', 'sensory_regulation',
                                  'social_skills_support', 'interest_integration'],
                'environmental_factors': ['reduced_sensory_stimulation', 'clear_transitions',
                                          'consistent_routines']
            },
            'Dyscalculia': {
                'primary_characteristics': [
                    'Difficulty understanding magnitude',
                    'Difficulty understanding quantity',
                    'Reduced ability to hold numerical information',
                    'Slow retrieval of math facts',
                    'Difficulty linking numbers to symbols',
                    'Challenges with spatial reasoning'
                ],
                'recommended_methodologies': [
                    'Gamified math activities', 'Multisensory Activities',
                    'Scaffolding', 'Concrete manipulatives'
                ],
                'support_needs': ['concrete_representations', 'visual_math_tools',
                                  'extra_processing_time', 'multi_modal_instruction'],
                'environmental_factors': ['low_anxiety_math_environment',
                                          'multiple_representation_formats']
            },
            'Dyslexia': {
                'primary_characteristics': [
                    'Increased cognitive load during reading tasks',
                    'Need for higher effort in self-monitoring',
                    'Risk of reduced self-efficacy',
                    'Frustration and stress during reading',
                    'Visual-spatial reasoning and creativity'
                ],
                'recommended_methodologies': [
                    'Multisensory structured literacy instruction',
                    'Text-to-speech tools', 'Scaffolding', 'Focus Tasks'
                ],
                'support_needs': ['reading_accommodations', 'alternative_text_formats',
                                  'metacognitive_support', 'self_efficacy_building'],
                'environmental_factors': ['accessible_fonts', 'reduced_reading_pressure',
                                          'strength_based_approach']
            },
            'Gifted': {
                'primary_characteristics': [
                    'Rapid learning pace',
                    'Intense curiosity and deep questioning',
                    'High cognitive abilities',
                    'Risk of boredom with routine tasks',
                    'Intense focus on areas of interest',
                    'Sensitivity to fairness and emotional nuances'
                ],
                'recommended_methodologies': [
                    'Self-Directed Exploration', 'Challenge-Based Learning',
                    'Project Based Learning', 'Inquiry-based learning', 'Debate'
                ],
                'support_needs': ['intellectual_challenge', 'depth_over_breadth',
                                  'autonomy_and_choice', 'creative_expression'],
                'environmental_factors': ['advanced_resources', 'flexible_pacing',
                                          'peer_intellectual_interaction']
            },
            'ForeignStudents': {
                'primary_characteristics': [
                    'Limited proficiency in the language of instruction',
                    'Difficulty expressing ideas verbally',
                    'Struggles to follow fast-paced instruction',
                    'Limited academic vocabulary',
                    'Cultural differences affect participation'
                ],
                'recommended_methodologies': [
                    'Cooperative Learning', 'Peer to Peer',
                    'Translanguaging activities', 'Linguistic scaffolds'
                ],
                'support_needs': ['dual_language_materials', 'visual_supports',
                                  'collaborative_glossaries', 'cultural_responsiveness'],
                'environmental_factors': ['culturally_responsive_environment',
                                          'peer_language_models', 'subtitle_support']
            },
        }

    def get_educational_context_type(self) -> str:
        return "inclusive_education"
