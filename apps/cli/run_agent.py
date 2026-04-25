#!/usr/bin/env python3
"""
Interactive Agent Testing Script

Test the Agentic GraphRAG pipeline with custom queries.
No hardcoded queries - prompts for input at runtime.

Run from the repo root:
    python apps/cli/run_agent.py
    python apps/cli/run_agent.py --domain neuro --language it
"""

import asyncio
import argparse
import logging
import sys
import os

# Phase 2 reorg: this file moved from <repo_root>/test_agent.py to
# <repo_root>/apps/cli/run_agent.py. Add the repo root to sys.path so the
# top-level "agent" package and other root modules remain importable.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if not os.path.isdir(os.path.join(PROJECT_ROOT, "agent")):
    print("❌ Error: cannot locate the 'agent' package at the repo root")
    print(f"   Expected: {os.path.join(PROJECT_ROOT, 'agent')}")
    sys.exit(1)

# Load environment variables (.env at repo root)
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Verify OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY not found in .env file")
    sys.exit(1)

from agent import AgentOrchestrator


def setup_logging(verbose: bool = False):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def print_header():
    """Print welcome header"""
    print("\n" + "=" * 60)
    print("🎓 Agentic GraphRAG - Interactive Testing")
    print("=" * 60)
    print("\nThis tool tests the multi-agent lesson planning pipeline.")
    print("Pipeline: Planner → Retriever → Writer → Critic\n")


def print_result(result):
    """Pretty print the result"""
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    
    print(f"\n✅ Success: {result.success}")
    print(f"📝 Approved: {result.approved}")
    print(f"🔄 Revisions: {result.revision_count}")
    print(f"📦 Nodes Used: {result.nodes_used}")
    print(f"💡 Recommendations: {result.recommendations_used}")
    
    if result.scores:
        print(f"\n📈 Scores: {result.scores}")
    
    if result.critique_summary:
        print(f"\n🔍 Critique: {result.critique_summary[:200]}...")
    
    if result.error:
        print(f"\n❌ Error: {result.error}")
    
    if result.lesson_plan:
        print("\n" + "=" * 60)
        print("📄 GENERATED LESSON PLAN")
        print("=" * 60)
        print(result.lesson_plan)
        
        # Offer to save
        save = input("\n💾 Save lesson plan to file? (y/n): ").strip().lower()
        if save == 'y':
            filename = f"lesson_plan_{asyncio.get_event_loop().time():.0f}.md"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(result.lesson_plan)
            print(f"✅ Saved to: {filename}")


async def run_interactive(domain: str, language: str, max_revisions: int):
    """Run interactive testing loop"""
    
    orchestrator = AgentOrchestrator(
        domain=domain,
        language=language,
        max_revisions=max_revisions
    )
    
    print(f"🔧 Configuration:")
    print(f"   Domain: {domain}")
    print(f"   Language: {language}")
    print(f"   Max Revisions: {max_revisions}")
    
    while True:
        print("\n" + "-" * 60)
        print("Enter your query (or 'quit' to exit, 'help' for examples):")
        query = input("📝 > ").strip()
        
        if not query:
            continue
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if query.lower() == 'help':
            print_examples()
            continue
        
        if query.lower() == 'switch':
            domain = input("New domain (neuro/udl): ").strip() or domain
            language = input("New language (it/en): ").strip() or language
            orchestrator = AgentOrchestrator(
                domain=domain,
                language=language,
                max_revisions=max_revisions
            )
            print(f"✅ Switched to domain={domain}, language={language}")
            continue
        
        print("\n⏳ Processing... (this may take 30-60 seconds)")
        print("   Pipeline: Planner → Retriever → Writer → Critic")
        
        try:
            result = await orchestrator.create_lesson_plan(query)
            print_result(result)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logging.exception("Full traceback:")


def print_examples():
    """Print example queries"""
    print("\n📚 Example Queries:")
    print("-" * 40)
    
    examples = [
        ("Lesson Creation", "Crea una lezione sulla motivazione per studenti con ADHD"),
        ("Activity Design", "Attività di 30 minuti sulla metacognizione per la scuola media"),
        ("Strategy Request", "Come posso introdurre il pensiero critico in classe?"),
        ("Comparison", "Qual è la differenza tra memoria a breve e lungo termine?"),
        ("Definition", "Cosa significa growth mindset?"),
        ("English Query", "Design a lesson on attention strategies for high school"),
    ]
    
    for category, query in examples:
        print(f"\n🔹 {category}:")
        print(f"   \"{query}\"")
    
    print("\n💡 Commands:")
    print("   'switch' - Change domain/language")
    print("   'quit'   - Exit the program")


async def run_single_query(query: str, domain: str, language: str, max_revisions: int):
    """Run a single query (non-interactive mode)"""
    
    orchestrator = AgentOrchestrator(
        domain=domain,
        language=language,
        max_revisions=max_revisions
    )
    
    print(f"\n🔧 Configuration: domain={domain}, language={language}")
    print(f"📝 Query: {query}")
    print("\n⏳ Processing...")
    
    result = await orchestrator.create_lesson_plan(query)
    print_result(result)


def main():
    parser = argparse.ArgumentParser(
        description="Test the Agentic GraphRAG lesson planning pipeline"
    )
    parser.add_argument(
        "--domain", "-d",
        type=str,
        default="neuro",
        choices=["neuro", "udl"],
        help="Knowledge domain (default: neuro)"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="it",
        choices=["it", "en"],
        help="Output language (default: it)"
    )
    parser.add_argument(
        "--max-revisions", "-r",
        type=int,
        default=2,
        help="Maximum revision cycles (default: 2)"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Single query to run (non-interactive mode)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    print_header()
    
    if args.query:
        # Non-interactive mode
        asyncio.run(run_single_query(
            args.query,
            args.domain,
            args.language,
            args.max_revisions
        ))
    else:
        # Interactive mode
        asyncio.run(run_interactive(
            args.domain,
            args.language,
            args.max_revisions
        ))


if __name__ == "__main__":
    main()

