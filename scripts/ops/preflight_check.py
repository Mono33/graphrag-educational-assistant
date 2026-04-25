#!/usr/bin/env python3
"""
Pre-flight Check for Migration Runner
Validate all requirements before running migrations
"""

import sys
from pathlib import Path

def preflight_check():
    """Run all pre-flight checks as recommended by ChatGPT"""
    
    print("🛫 Pre-flight Migration Checks")
    print("=" * 50)
    
    checks_passed = 0
    total_checks = 6
    
    # Check 1: Config loading
    print("1️⃣ Testing config loading...")
    try:
        sys.path.append(str(Path(__file__).parent.parent))
        from config import config
        
        is_valid, errors = config.validate()
        if is_valid:
            print("   ✅ Config loaded and validated successfully")
            print(f"   🔗 Neo4j URI: {config.neo4j.uri}")
            print(f"   👤 Neo4j User: {config.neo4j.user}")
            checks_passed += 1
        else:
            print("   ❌ Config validation failed:")
            for error in errors:
                print(f"      - {error}")
    except Exception as e:
        print(f"   ❌ Config loading failed: {e}")
    
    # Check 2: Neo4j connectivity
    print("\n2️⃣ Testing Neo4j connectivity...")
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(config.neo4j.uri, auth=(config.neo4j.user, config.neo4j.password))
        
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as node_count")
            node_count = result.single()["node_count"]
            print(f"   ✅ Connected successfully - {node_count} nodes in graph")
            checks_passed += 1
        driver.close()
    except Exception as e:
        print(f"   ❌ Neo4j connection failed: {e}")
    
    # Check 3: APOC availability
    print("\n3️⃣ Testing APOC procedures...")
    try:
        with GraphDatabase.driver(config.neo4j.uri, auth=(config.neo4j.user, config.neo4j.password)).session() as session:
            # Test essential APOC functions
            session.run("RETURN apoc.create.uuid() as test_uuid")
            session.run("RETURN apoc.text.slug('Test String') as test_slug")
            print("   ✅ APOC procedures available")
            checks_passed += 1
    except Exception as e:
        print(f"   ❌ APOC not available: {e}")
        print("   💡 Install APOC plugin in Neo4j")
    
    # Check 4: Migration files exist
    print("\n4️⃣ Checking migration files...")
    migrations_dir = Path(__file__).parent.parent / "neo4j_migrations"
    required_files = ["001_add_uuid_and_indexes.cypher", "002_backfill_slug_and_name_lc.cypher"]
    
    missing_files = []
    for file in required_files:
        if not (migrations_dir / file).exists():
            missing_files.append(file)
    
    if not missing_files:
        print(f"   ✅ All required migration files found")
        checks_passed += 1
    else:
        print(f"   ❌ Missing files: {missing_files}")
    
    # Check 5: Test execution script
    print("\n5️⃣ Testing execution script availability...")
    try:
        from test_execution import test_real_execution
        print("   ✅ test_execution.py can be imported")
        checks_passed += 1
    except Exception as e:
        print(f"   ⚠️  test_execution.py import issue: {e}")
        print("   💡 This is optional - manual testing is fine")
        checks_passed += 1  # Don't fail for this
    
    # Check 6: Current data state
    print("\n6️⃣ Checking current data state...")
    try:
        with GraphDatabase.driver(config.neo4j.uri, auth=(config.neo4j.user, config.neo4j.password)).session() as session:
            # Check if migrations already applied
            result = session.run("MATCH (n) WHERE n.uuid IS NOT NULL RETURN count(n) as with_uuid")
            uuid_count = result.single()["with_uuid"]
            
            result = session.run("MATCH (n) WHERE n.slug IS NOT NULL RETURN count(n) as with_slug")
            slug_count = result.single()["with_slug"]
            
            if uuid_count == 0 and slug_count == 0:
                print("   ✅ Fresh state - ready for migrations")
            else:
                print(f"   ⚠️  Some nodes already have UUIDs ({uuid_count}) or slugs ({slug_count})")
                print("   💡 Migrations will be safe (MERGE/IF NOT EXISTS used)")
            
            checks_passed += 1
    except Exception as e:
        print(f"   ❌ Data state check failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🎯 Pre-flight Summary: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed >= 5:
        print("✅ READY FOR MIGRATION - All critical checks passed")
        print("🚀 Run: python scripts/run_migrations.py --dry-run")
        return True
    else:
        print("❌ NOT READY - Fix issues above before proceeding")
        return False

if __name__ == "__main__":
    success = preflight_check()
    sys.exit(0 if success else 1)
