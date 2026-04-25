#!/usr/bin/env python3
"""
Test script for Hybrid Translation (Solution 3)
Tests dictionary coverage and OpenAI fallback
"""

from multilingual_text2cypher import MultilingualText2Cypher
import logging

# Enable logging to see the hybrid logic
logging.basicConfig(
    level=logging.INFO, 
    format='%(levelname)s - %(message)s'
)

# Initialize translator
translator = MultilingualText2Cypher()

# Test queries (same as Pool 1)
test_queries = [
    'Qual è la differenza tra motivazione intrinseca ed estrinseca?',
    'Come lo stress influisce sulla memoria di lavoro?',
    'Cosa significa mentalità di crescita in parole semplici?',
    'Quali sono le principali caratteristiche di una mentalità fissa?'
]

print('=' * 80)
print('🧪 TESTING HYBRID TRANSLATION (Solution 3)')
print('=' * 80)
print('\nHow it works:')
print('1. Try dictionary translation (fast, free)')
print('2. Calculate coverage')
print('3. If coverage < 50%, use OpenAI (accurate, ~$0.0001 per query)')
print('=' * 80)

for i, query in enumerate(test_queries, 1):
    print(f'\n📝 Query {i}: {query}')
    print('-' * 80)
    
    # Translate with hybrid approach
    enhanced = translator.enhance_italian_query(query, domain='neuro')
    
    # Remove context prefix for display
    if ': ' in enhanced:
        display = enhanced.split(': ', 1)[1]
    else:
        display = enhanced
    
    print(f'✅ Translated: {display}')
    print()

print('=' * 80)
print('✅ Test Complete!')
print('\n📊 Expected Results:')
print('  Query 1: Dictionary (60-70% coverage) ✅')
print('  Query 2: Dictionary (60-70% coverage) ✅')
print('  Query 3: OpenAI Fallback (<50% coverage) 🔄')
print('  Query 4: Dictionary (50-60% coverage) ✅')
print('=' * 80)

translator.close()

