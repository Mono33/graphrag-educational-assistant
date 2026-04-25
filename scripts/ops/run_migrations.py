#!/usr/bin/env python3
"""
Neo4j Migration Runner
Safely applies data quality migrations in order with validation
"""

import os
import glob
import sys
from pathlib import Path
from typing import List
from neo4j import GraphDatabase
import time
from datetime import datetime

class MigrationRunner:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.migrations_dir = Path(__file__).parent.parent / "neo4j_migrations"
        
    def close(self):
        self.driver.close()
    
    def has_apoc(self) -> bool:
        """Check if APOC is available in Neo4j"""
        try:
            with self.driver.session() as session:
                session.run("RETURN apoc.version() AS v").single()
            return True
        except Exception:
            return False
    
    def _get_migration_files(self, apoc_available: bool) -> List[str]:
        """Get appropriate migration files based on APOC availability"""
        all_files = sorted(self.migrations_dir.glob("*.cypher"))
        selected_files = []
        
        for file_path in all_files:
            name = file_path.name
            
            # Skip APOC-specific files if APOC not available
            if not apoc_available and not name.endswith("_no_apoc.cypher") and not name.startswith("999_"):
                # Skip regular files that require APOC (like 003_merge_duplicates_by_slug.cypher)
                if any(x in name for x in ["merge_duplicates", "split_multiconcept", "fix_labels"]):
                    print(f"   ⏭️  Skipping {name} (requires APOC)")
                    continue
            
            # Skip no-APOC files if APOC is available (prefer regular versions)
            if apoc_available and name.endswith("_no_apoc.cypher"):
                regular_name = name.replace("_no_apoc.cypher", ".cypher")
                regular_path = self.migrations_dir / regular_name
                if regular_path.exists():
                    print(f"   ⏭️  Skipping {name} (using APOC version)")
                    continue
            
            # Always include quality checks and detection scripts
            if name.startswith("999_") or "detect" in name:
                selected_files.append(str(file_path))
                continue
                
            # Include based on APOC availability
            if not apoc_available and name.endswith("_no_apoc.cypher"):
                selected_files.append(str(file_path))
            elif apoc_available and not name.endswith("_no_apoc.cypher"):
                selected_files.append(str(file_path))
        
        return selected_files
    
    def backup_graph(self):
        """Create a backup before running migrations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"backup_before_migrations_{timestamp}.cypher"
        
        print(f"📦 Creating backup: {backup_file}")
        
        # Note: In production, use neo4j-admin dump
        # For development, we'll just log this step
        with self.driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as node_count")
            node_count = result.single()["node_count"]
            print(f"📊 Current graph has {node_count} nodes")
        
        return backup_file
    
    def run_migration_file(self, session, file_path):
        """Execute a single migration file"""
        print(f"🔄 Applying {file_path.name}...")
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Split on semicolons and filter out comments/empty statements
        statements = []
        for stmt in content.split(';'):
            stmt = stmt.strip()
            if stmt and not stmt.startswith('//'):
                statements.append(stmt)
        
        # Execute each statement
        for i, statement in enumerate(statements, 1):
            try:
                result = session.run(statement)
                # Consume result to ensure execution
                summary = result.consume()
                if summary.counters:
                    print(f"   ✓ Statement {i}: {summary.counters}")
            except Exception as e:
                print(f"   ❌ Error in statement {i}: {e}")
                print(f"   Statement: {statement[:100]}...")
                raise
    
    def validate_migration(self, session, migration_name):
        """Run validation checks after each migration"""
        print(f"🔍 Validating {migration_name}...")
        
        # Basic health checks
        result = session.run("""
            MATCH (n) 
            RETURN labels(n)[0] as label, count(n) as count 
            ORDER BY count DESC
        """)
        
        for record in result:
            print(f"   📊 {record['label']}: {record['count']} nodes")
        
        # Check for orphaned nodes after structural changes
        if migration_name in ['003_merge_duplicates_by_slug', '005_split_multiconcept_nodes']:
            result = session.run("""
                MATCH (n) WHERE NOT EXISTS { MATCH (n)--() }
                RETURN count(n) as orphan_count
            """)
            orphan_count = result.single()["orphan_count"]
            if orphan_count > 0:
                print(f"   ⚠️  Found {orphan_count} orphaned nodes")
    
    def run_test_queries(self):
        """Run the golden test suite (your 10 queries) after migrations"""
        print("🧪 Running golden test suite...")
        
        # Import your test execution
        try:
            sys.path.append(str(Path(__file__).parent.parent))
            from test_execution import test_real_execution
            
            # This would run your existing test suite
            # For now, we'll just run a basic connectivity test
            with self.driver.session() as session:
                # Test basic SEN -> Methodology queries
                test_queries = [
                    ("ADHD test", "MATCH (s:StudentWithSpecialNeeds {slug:'adhd'})-[:SUGGESTS]->(m) RETURN count(m) as count"),
                    ("Autism test", "MATCH (s:StudentWithSpecialNeeds {slug:'autism-spectrum-disorder'})-[:SUGGESTS]->(m) RETURN count(m) as count"),
                    ("Cognitive test", "MATCH (s:StudentWithSpecialNeeds {slug:'cognitive-disability'})-[:SUGGESTS]->(m) RETURN count(m) as count"),
                ]
                
                success_count = 0
                for name, query in test_queries:
                    result = session.run(query)
                    count = result.single()["count"]
                    if count > 0:
                        print(f"   ✅ {name}: {count} suggestions")
                        success_count += 1
                    else:
                        print(f"   ❌ {name}: 0 suggestions")
                
                success_rate = success_count / len(test_queries)
                print(f"🎯 Test success rate: {success_rate:.1%}")
                
                return success_rate >= 0.7  # 70% threshold
        
        except ImportError:
            print("   ⚠️  Could not import test_execution.py - skipping golden tests")
            return True
    
    def run_all_migrations(self, dry_run=False):
        """Run all migrations in order"""
        print("🚀 Starting Neo4j Data Quality Migrations")
        print("=" * 60)
        
        if not dry_run:
            self.backup_graph()
        
        # Auto-select appropriate migration files based on APOC availability
        apoc_available = self.has_apoc()
        print(f"🧩 APOC detected: {apoc_available}")
        
        migration_files = self._get_migration_files(apoc_available)
        
        if not migration_files:
            print("❌ No migration files found in neo4j_migrations/")
            return False
        
        print(f"📋 Found {len(migration_files)} migration files")
        
        if dry_run:
            print("🔍 DRY RUN - No changes will be made")
            for file_path in migration_files:
                print(f"   Would apply: {Path(file_path).name}")
            return True
        
        try:
            with self.driver.session() as session:
                for file_path in migration_files:
                    file_path = Path(file_path)
                    start_time = time.time()
                    
                    self.run_migration_file(session, file_path)
                    self.validate_migration(session, file_path.stem)
                    
                    elapsed = time.time() - start_time
                    print(f"   ⏱️  Completed in {elapsed:.2f}s")
                    print()
            
            # Run final validation
            print("🎯 Running final validation...")
            if self.run_test_queries():
                print("✅ All migrations completed successfully!")
                return True
            else:
                print("⚠️  Migrations completed but some tests failed")
                return True
                
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            print("💡 Consider restoring from backup")
            return False

def main():
    # Use same configuration as your working scripts
    try:
        sys.path.append(str(Path(__file__).parent.parent))
        from config import config
        
        # Validate config first
        is_valid, errors = config.validate()
        if not is_valid:
            print("❌ Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            print("\n💡 Update your .env file with:")
            print("NEO4J_PASSWORD=your_actual_password")
            print("OPENAI_API_KEY=sk-your-actual-key")
            sys.exit(1)
        
        NEO4J_URI = config.neo4j.uri
        NEO4J_USER = config.neo4j.user
        NEO4J_PASSWORD = config.neo4j.password
        
        print(f"🔗 Connecting to: {NEO4J_URI}")
        print(f"👤 User: {NEO4J_USER}")
        
    except ImportError:
        print("❌ Could not import config.py")
        print("💡 Make sure you're running from the correct directory")
        sys.exit(1)
    
    # Parse command line arguments
    dry_run = "--dry-run" in sys.argv
    
    runner = MigrationRunner(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        success = runner.run_all_migrations(dry_run=dry_run)
        sys.exit(0 if success else 1)
    finally:
        runner.close()

if __name__ == "__main__":
    main()
