"""
Quick test for Intent Detection in Phase 2 Agent Mode.
Tests that the Planner correctly identifies query intents.
"""

import asyncio

from aix.agent.agents.planner_agent import PlannerAgent


async def test_intent_detection():
    planner = PlannerAgent()

    test_queries = [
        ("Crea una lezione sulla memoria di lavoro", "lesson_creation"),
        ("Cos'è la neuroplasticità?", "definition"),
        ("Qual è la differenza tra memoria a breve e lungo termine?", "comparison"),
        ("Come funziona l'attenzione selettiva?", "explanation"),
        ("Quali strategie per studenti con ADHD?", "recommendation"),
        ("Elenca i tipi di memoria", "list"),
        ("Attività di 30 minuti sulla metacognizione", "activity_design"),
    ]

    print("=" * 60)
    print("Testing Intent Detection (Phase 2)")
    print("=" * 60)
    print()

    correct = 0
    total = len(test_queries)

    for query, expected in test_queries:
        plan = await planner.plan(query, domain="neuro", language="it")
        is_correct = plan.query_intent == expected
        status = "✅" if is_correct else "❌"

        if is_correct:
            correct += 1

        print(f'{status} Query: "{query}"')
        print(f'   Expected: {expected}')
        print(f'   Got: {plan.query_intent} (confidence: {plan.intent_confidence})')
        if plan.key_concepts:
            print(f'   Concepts: {", ".join(plan.key_concepts[:3])}')
        print()

    print("=" * 60)
    print(f"Results: {correct}/{total} correct ({100*correct/total:.0f}%)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_intent_detection())

