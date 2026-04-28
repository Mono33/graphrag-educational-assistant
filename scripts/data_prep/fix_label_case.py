#!/usr/bin/env python3
"""Fix case-variant duplicate labels in Neo4j (one-time remediation)"""

from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()
uri = os.getenv('NEO4J_URI')
user = os.getenv('NEO4J_USER')
password = os.getenv('NEO4J_PASSWORD')

driver = GraphDatabase.driver(uri, auth=(user, password))

RELABEL_PAIRS = [
    ('Teachingpractices', 'TeachingPractices'),
    ('Executivefunctions', 'ExecutiveFunctions'),
    ('Cognitiveneuroscience', 'CognitiveNeuroscience'),
    ('Learningoutcomes', 'LearningOutcomes'),
    ('Cognitiveload', 'CognitiveLoad'),
    ('Emotionalregulation', 'EmotionalRegulation'),
    ('Sociallearning', 'SocialLearning'),
    ('Growthmindset', 'GrowthMindset'),
    ('Educationalmyths', 'EducationalMyths'),
]

print("=" * 60)
print("  FIX LABEL CASE VARIANTS IN NEO4J")
print("=" * 60)

total_fixed = 0
with driver.session() as session:
    for old_label, new_label in RELABEL_PAIRS:
        result = session.run(
            f"MATCH (n:{old_label} {{domain: $domain}}) RETURN count(n) as cnt",
            domain="neuro"
        )
        count = result.single()['cnt']
        if count > 0:
            session.run(
                f"MATCH (n:{old_label} {{domain: $domain}}) REMOVE n:{old_label} SET n:{new_label}",
                domain="neuro"
            )
            print(f"  Fixed: {old_label} -> {new_label} ({count} nodes)")
            total_fixed += count
        else:
            print(f"  Skip: {old_label} (0 nodes)")

print(f"\nTotal nodes relabeled: {total_fixed}")
print("Done!")
driver.close()
