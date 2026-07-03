#!/usr/bin/env python3
"""
export_neo4j_data.py - Export Neo4j Knowledge Graph Data
Exports all nodes and relationships from a Neo4j instance to JSON format
for migration to another Neo4j instance.
"""

import argparse
import json
from datetime import datetime
from typing import Any

from neo4j import GraphDatabase


def export_nodes(session, domain: str = None) -> list[dict[str, Any]]:
    """Export all nodes, optionally filtered by domain"""
    if domain and domain != "all":
        query = """
        MATCH (n)
        WHERE n.domain = $domain
        RETURN n, labels(n) as labels, elementId(n) as id
        """
        result = session.run(query, domain=domain)
    else:
        query = """
        MATCH (n)
        RETURN n, labels(n) as labels, elementId(n) as id
        """
        result = session.run(query)

    nodes = []
    for record in result:
        node = dict(record["n"])
        node["_labels"] = record["labels"]
        node["_export_id"] = record["id"]
        nodes.append(node)

    return nodes


def export_relationships(session, domain: str = None) -> list[dict[str, Any]]:
    """Export all relationships, optionally filtered by domain"""
    if domain and domain != "all":
        query = """
        MATCH (a)-[r]->(b)
        WHERE a.domain = $domain OR b.domain = $domain
        RETURN
            a.name as source_name,
            a.domain as source_domain,
            labels(a) as source_labels,
            type(r) as rel_type,
            properties(r) as rel_props,
            b.name as target_name,
            b.domain as target_domain,
            labels(b) as target_labels
        """
        result = session.run(query, domain=domain)
    else:
        query = """
        MATCH (a)-[r]->(b)
        RETURN
            a.name as source_name,
            a.domain as source_domain,
            labels(a) as source_labels,
            type(r) as rel_type,
            properties(r) as rel_props,
            b.name as target_name,
            b.domain as target_domain,
            labels(b) as target_labels
        """
        result = session.run(query)

    relationships = []
    for record in result:
        rel = {
            "source": {
                "name": record["source_name"],
                "domain": record["source_domain"],
                "labels": record["source_labels"]
            },
            "target": {
                "name": record["target_name"],
                "domain": record["target_domain"],
                "labels": record["target_labels"]
            },
            "type": record["rel_type"],
            "properties": dict(record["rel_props"]) if record["rel_props"] else {}
        }
        relationships.append(rel)

    return relationships


def export_to_json(uri: str, user: str, password: str, output_file: str, domain: str = None):
    """Export entire graph to JSON file"""
    print(f"🔗 Connecting to Neo4j: {uri}")
    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session() as session:
            print(f"📤 Exporting nodes{f' (domain: {domain})' if domain else ''}...")
            nodes = export_nodes(session, domain)
            print(f"   Found {len(nodes)} nodes")

            print(f"📤 Exporting relationships{f' (domain: {domain})' if domain else ''}...")
            relationships = export_relationships(session, domain)
            print(f"   Found {len(relationships)} relationships")

            # Create export structure
            export_data = {
                "metadata": {
                    "export_date": datetime.now().isoformat(),
                    "source_uri": uri,
                    "domain_filter": domain,
                    "total_nodes": len(nodes),
                    "total_relationships": len(relationships)
                },
                "nodes": nodes,
                "relationships": relationships
            }

            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

            print("\n✅ Export complete!")
            print(f"   📁 File: {output_file}")
            print(f"   📊 Nodes: {len(nodes)}")
            print(f"   🔗 Relationships: {len(relationships)}")

    finally:
        driver.close()


def main():
    parser = argparse.ArgumentParser(description="Export Neo4j graph data to JSON")
    parser.add_argument("--uri", required=True, help="Neo4j URI (source)")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", required=True, help="Neo4j password")
    parser.add_argument("--output", default="graph_export.json", help="Output JSON file")
    parser.add_argument("--domain", help="Filter by domain (neuro, udl, or all)")

    args = parser.parse_args()

    export_to_json(args.uri, args.user, args.password, args.output, args.domain)


if __name__ == "__main__":
    main()

