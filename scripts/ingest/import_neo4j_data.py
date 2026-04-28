#!/usr/bin/env python3
"""
import_neo4j_data.py - Import Neo4j Knowledge Graph Data
Imports nodes and relationships from a JSON export file to a Neo4j instance.
"""

import json
import argparse
from datetime import datetime
from neo4j import GraphDatabase
from typing import Dict, List, Any


def clear_domain(session, domain: str):
    """Clear all nodes for a specific domain"""
    query = """
    MATCH (n {domain: $domain})
    DETACH DELETE n
    """
    result = session.run(query, domain=domain)
    summary = result.consume()
    print(f"🗑️  Cleared domain '{domain}' - deleted {summary.counters.nodes_deleted} nodes")


def clear_all(session):
    """Clear entire database"""
    query = "MATCH (n) DETACH DELETE n"
    result = session.run(query)
    summary = result.consume()
    print(f"🗑️  Cleared entire database - deleted {summary.counters.nodes_deleted} nodes")


def create_node(session, node: Dict[str, Any]) -> bool:
    """Create a single node with its labels and properties"""
    labels = node.get("_labels", ["Concept"])
    
    # Remove internal export fields
    props = {k: v for k, v in node.items() if not k.startswith("_")}
    
    # Build label string
    label_str = ":".join(labels)
    
    # Create node with MERGE to avoid duplicates
    query = f"""
    MERGE (n:{label_str} {{name: $name, domain: $domain}})
    SET n += $props
    """
    
    try:
        session.run(query, 
                   name=props.get("name", "Unknown"),
                   domain=props.get("domain", "unknown"),
                   props=props)
        return True
    except Exception as e:
        print(f"⚠️  Error creating node {props.get('name')}: {e}")
        return False


def create_relationship(session, rel: Dict[str, Any]) -> bool:
    """Create a single relationship between nodes"""
    source = rel["source"]
    target = rel["target"]
    rel_type = rel["type"]
    rel_props = rel.get("properties", {})
    
    # Match source and target by name and domain, then create relationship
    query = f"""
    MATCH (a {{name: $source_name, domain: $source_domain}})
    MATCH (b {{name: $target_name, domain: $target_domain}})
    MERGE (a)-[r:{rel_type}]->(b)
    SET r += $rel_props
    """
    
    try:
        session.run(query,
                   source_name=source["name"],
                   source_domain=source.get("domain"),
                   target_name=target["name"],
                   target_domain=target.get("domain"),
                   rel_props=rel_props)
        return True
    except Exception as e:
        print(f"⚠️  Error creating relationship {source['name']} -[{rel_type}]-> {target['name']}: {e}")
        return False


def import_from_json(uri: str, user: str, password: str, input_file: str, 
                     clear: bool = False, clear_domain_name: str = None):
    """Import graph data from JSON file"""
    
    # Load export file
    print(f"📂 Loading export file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data.get("metadata", {})
    nodes = data.get("nodes", [])
    relationships = data.get("relationships", [])
    
    print(f"   📊 Source: {metadata.get('source_uri', 'unknown')}")
    print(f"   📅 Export date: {metadata.get('export_date', 'unknown')}")
    print(f"   📦 Nodes to import: {len(nodes)}")
    print(f"   🔗 Relationships to import: {len(relationships)}")
    
    # Connect to target Neo4j
    print(f"\n🔗 Connecting to target Neo4j: {uri}")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
        with driver.session() as session:
            # Clear if requested
            if clear:
                clear_all(session)
            elif clear_domain_name:
                clear_domain(session, clear_domain_name)
            
            # Import nodes
            print(f"\n📥 Importing {len(nodes)} nodes...")
            node_success = 0
            for i, node in enumerate(nodes, 1):
                if create_node(session, node):
                    node_success += 1
                if i % 100 == 0:
                    print(f"   Progress: {i}/{len(nodes)} nodes...")
            print(f"   ✅ Imported {node_success}/{len(nodes)} nodes")
            
            # Import relationships
            print(f"\n📥 Importing {len(relationships)} relationships...")
            rel_success = 0
            for i, rel in enumerate(relationships, 1):
                if create_relationship(session, rel):
                    rel_success += 1
                if i % 100 == 0:
                    print(f"   Progress: {i}/{len(relationships)} relationships...")
            print(f"   ✅ Imported {rel_success}/{len(relationships)} relationships")
            
            print(f"\n✅ Import complete!")
            print(f"   📊 Nodes: {node_success}/{len(nodes)}")
            print(f"   🔗 Relationships: {rel_success}/{len(relationships)}")
            
    finally:
        driver.close()


def main():
    parser = argparse.ArgumentParser(description="Import Neo4j graph data from JSON")
    parser.add_argument("--uri", required=True, help="Neo4j URI (target)")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", required=True, help="Neo4j password")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--clear", action="store_true", help="Clear entire database before import")
    parser.add_argument("--clear-domain", help="Clear specific domain before import")
    
    args = parser.parse_args()
    
    import_from_json(args.uri, args.user, args.password, args.input, 
                     args.clear, args.clear_domain)


if __name__ == "__main__":
    main()

