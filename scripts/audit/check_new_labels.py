#!/usr/bin/env python3
"""
Check statistics for ALL labels in neuro_domain.py get_node2vec_weights()
Verifies that weights are appropriate based on actual graph data.
"""

from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import math

load_dotenv()
uri = os.getenv('NEO4J_URI')
user = os.getenv('NEO4J_USER')
password = os.getenv('NEO4J_PASSWORD')

driver = GraphDatabase.driver(uri, auth=(user, password))

# ALL labels from neuro_domain.py get_node2vec_weights()
all_labels = [
    # TOP 10 MOST FREQUENT LABELS
    ('Attention', 2.2),
    ('CriticalThinking', 2.0),
    ('ExtrinsicMotivation', 1.9),
    ('ExecutiveFunctions', 2.1),
    ('IntrinsicMotivation', 2.0),
    ('LearningOutcomes', 1.8),
    ('TeachingPractices', 1.8),
    ('LearningDevelopment', 1.7),
    ('NegativeStressDistress', 1.7),
    ('Motivation', 1.6),
    
    # HUB NODES
    ('CognitiveFlexibility', 2.0),
    ('KnowledgeConstructionAttention', 1.9),
    ('PrefrontalCortexActivation', 1.9),
    ('OptimalAttentionalNetworkActivation', 1.8),
    
    # AUTHORITY NODES
    ('Creativity', 1.8),
    ('Memory', 1.7),
    ('MemoryEncoding', 1.6),
    ('MemorySystems', 1.6),
    
    # CRITICAL COGNITIVE PROCESSES
    ('WorkingMemory', 1.7),
    ('Metacognition', 1.6),
    ('SelfRegulation', 1.5),
    ('CognitiveControl', 1.6),
    ('CognitiveProcesses', 1.6),
    
    # AFFECTIVE & MOTIVATIONAL
    ('EmotionalRegulation', 1.6),
    ('EmotionalWellBeing', 1.4),
    ('PositiveEmotions', 1.6),
    ('NegativeEmotions', 1.5),
    ('AffectiveProcesses', 1.5),
    
    # MINDSET & GROWTH
    ('GrowthMindset', 1.7),
    ('FixedMindset', 1.5),
    ('Mindset', 1.6),
    
    # STRESS & COPING
    ('PositiveStressEustress', 1.6),
    ('StressResponse', 1.5),
    ('LongTermGrowth', 1.5),
    ('LongTermDecline', 1.4),
    ('AdaptiveCoping', 1.4),
    ('MaladaptiveCoping', 1.4),
    
    # SOCIAL & COMMUNICATION
    ('SocialCognition', 1.5),
    ('SocialLearning', 1.4),
    ('Communication', 1.4),
    
    # EDUCATIONAL OUTCOMES
    ('LearningEngagement', 1.5),
    ('LearningPerformance', 1.6),
    ('EducationalSupport', 1.5),
    
    # ADDITIONAL IMPORTANT
    ('HigherOrderThinking', 1.5),
    ('LowerOrderThinking', 1.3),
    ('ProblemSolving', 1.4),
    ('LongTermMemory', 1.5),
    ('PersonalGrowth', 1.4),
    ('Strengths', 1.4),
    ('CognitiveStrengths', 1.4),
    ('ReflectiveThinking', 1.1),
    ('Consolidation', 1.1),
    ('AffectiveMotivationalModulation', 1.2),  # Correct label
    ('AffectiveMotivationalProcesses', 1.2),   # Additional label
    ('BrainAdaptability', 1.1),
    ('Vulnerability', 1.3),
    ('Resilience', 1.5),
    ('CognitiveBias', 1.3),
    
    # NEW LABELS (Dec 2025)
    ('Learningstrategies', 1.5),
    ('LearningStrategies', 1.1),
    ('DistributedPracticeEffect', 1.1),
    ('LearningProgress', 1.2),  # Correct label (was LearningProcess)
    ('LongTermLearning', 1.2),
    ('Assessment', 1.5),
    ('Evaluation', 1.1),
    ('KnowledgeOfCognition', 1.1),
    ('HemisphericSpecialization', 1.1),
    ('Educationalmyths', 1.1),
    ('CognitiveNeuroscience', 1.2),
    ('NeurodevelopmentalLinks', 1.2),
    ('Neuroplasticity', 1.2),
    ('Brainfunction', 1.1),  # Note: lowercase 'f'
    ('Comorbidities', 1.1),
    ('LearningStyles', 1.1),
]

print('=' * 100)
print('Label Statistics for ALL labels in neuro_domain.py')
print('=' * 100)
print(f'{"Label":<35} {"Nodes":<8} {"In-Deg":<8} {"Out-Deg":<10} {"Suggested":<10} {"Current":<10} {"Status"}')
print('=' * 100)

issues = []
good = []
not_in_graph = []

with driver.session() as session:
    for label, current_weight in all_labels:
        # Count nodes
        count_query = f"MATCH (n:{label}) WHERE n.domain = 'neuro' RETURN count(n) as cnt"
        try:
            count_result = session.run(count_query)
            count = count_result.single()['cnt']
        except Exception as e:
            count = 0
        
        if count > 0:
            # Count relationships
            in_query = f"MATCH (n:{label})<-[r]-() WHERE n.domain = 'neuro' RETURN count(r) as cnt"
            out_query = f"MATCH (n:{label})-[r]->() WHERE n.domain = 'neuro' RETURN count(r) as cnt"
            
            in_deg = session.run(in_query).single()['cnt']
            out_deg = session.run(out_query).single()['cnt']
            
            # Calculate suggested weight
            connectivity = in_deg + out_deg
            suggested = 1.0 + math.log10(max(count, 1)) * 0.3 + math.log10(max(connectivity, 1) + 1) * 0.2
            suggested = min(max(suggested, 1.0), 2.5)
            
            # Compare with current weight
            diff = abs(current_weight - suggested)
            if diff <= 0.2:
                status = "✅ Good"
                good.append((label, count, suggested, current_weight))
            elif current_weight > suggested + 0.3:
                status = "⚠️ Too high"
                issues.append((label, count, suggested, current_weight, "too high"))
            elif current_weight < suggested - 0.3:
                status = "⚠️ Too low"
                issues.append((label, count, suggested, current_weight, "too low"))
            else:
                status = "⚡ Adjust"
                issues.append((label, count, suggested, current_weight, "adjust"))
            
            print(f'{label:<35} {count:<8} {in_deg:<8} {out_deg:<10} {suggested:<10.2f} {current_weight:<10} {status}')
        else:
            not_in_graph.append(label)
            print(f'{label:<35} {0:<8} {"N/A":<8} {"N/A":<10} {"1.0":<10} {current_weight:<10} ❌ Not in graph')

driver.close()

print('=' * 100)
print(f'\nSUMMARY:')
print(f'  ✅ Good weights: {len(good)}')
print(f'  ⚠️ Need adjustment: {len(issues)}')
print(f'  ❌ Not in graph: {len(not_in_graph)}')

if issues:
    print(f'\nLabels needing adjustment:')
    for label, count, suggested, current, reason in issues:
        print(f'  - {label}: current={current}, suggested={suggested:.2f} ({reason})')

if not_in_graph:
    print(f'\nLabels NOT in graph (can be removed from weights):')
    for label in not_in_graph:
        print(f'  - {label}')

print('\nWeight Formula: 1.0 + log10(nodes)*0.3 + log10(connectivity+1)*0.2')
print('Tolerance: ±0.2 from suggested weight is considered "Good"')
