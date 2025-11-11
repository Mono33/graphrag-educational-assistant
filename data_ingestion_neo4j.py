#!/usr/bin/env python3
"""
data_ingestion_neo4j.py - Import data from concepts_neo4j.json into Neo4j database
"""

import json
import os
from neo4j import GraphDatabase
import argparse

class Neo4jImporter:
    """Class to handle Neo4j data ingestion operations"""
    
    def __init__(self, uri, user, password):
        """Initialize connection to Neo4j"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        """Close the Neo4j connection"""
        self.driver.close()
    
    def create_node(self, tx, label, properties):
        """Create or update a node with given label and properties (using MERGE to avoid duplicates)"""
        # MERGE on 'id' property to avoid duplicates
        # ON CREATE SET: set all properties when creating new node
        # ON MATCH SET: update all properties when node already exists
        props_set_string = ", ".join([f"n.{k} = ${k}" for k in properties.keys()])
        
        query = f"""
        MERGE (n:{label} {{id: $id}})
        ON CREATE SET {props_set_string}
        ON MATCH SET {props_set_string}
        RETURN n.id as node_id, elementId(n) as element_id
        """
        result = tx.run(query, **properties)
        record = result.single()
        return record["element_id"], record.get("node_id")
    
    def create_relationship(self, tx, from_id, to_id, rel_type, properties=None):
        """Create or update a relationship between two nodes (using MERGE to avoid duplicates)"""
        if properties:
            props_string = ", ".join([f"{k}: ${k}" for k in properties.keys()])
            query = f"""
            MATCH (a), (b) 
            WHERE elementId(a) = $from_id AND elementId(b) = $to_id 
            MERGE (a)-[r:{rel_type}]->(b)
            SET {', '.join([f"r.{k} = ${k}" for k in properties.keys()])}
            """
            tx.run(query, from_id=from_id, to_id=to_id, **properties)
        else:
            query = f"""
            MATCH (a), (b) 
            WHERE elementId(a) = $from_id AND elementId(b) = $to_id 
            MERGE (a)-[:{rel_type}]->(b)
            """
            tx.run(query, from_id=from_id, to_id=to_id)
    
    def find_node_by_id(self, tx, node_id):
        """Find a node by its ID property"""
        query = "MATCH (n) WHERE n.id = $node_id RETURN elementId(n) as element_id"
        result = tx.run(query, node_id=node_id)
        record = result.single()
        return record["element_id"] if record else None
    
    def clear_database(self):
        """Clear all nodes and relationships in the database"""
        with self.driver.session() as session:
            session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))
            print("✅ Database cleared - all nodes and relationships deleted")
    
    def clear_domain(self, domain):
        """Clear all nodes and relationships for a specific domain
        
        Args:
            domain: Domain tag to clear (e.g., 'udl', 'neuro', 'game')
        """
        def delete_domain_nodes(tx, domain):
            """Delete all nodes for a domain and return count"""
            result = tx.run(
                "MATCH (n {domain: $domain}) DETACH DELETE n RETURN count(n) as deleted_count",
                domain=domain
            )
            record = result.single()
            return record["deleted_count"] if record else 0
        
        with self.driver.session() as session:
            deleted = session.execute_write(delete_domain_nodes, domain)
            print(f"✅ Cleared domain '{domain}' - deleted {deleted} nodes")
    
    def import_data(self, data_file, domain=None, log_file=None):
        """Import data from a JSON file into Neo4j with detailed logging
        
        Args:
            data_file: Path to the JSON file to import
            domain: Domain tag to add to all nodes (e.g., 'udl', 'neuro', 'game')
            log_file: Optional file handle to write logs to
        """
        import time
        from collections import Counter
        
        def log(message):
            """Print and optionally write to log file"""
            print(message)
            if log_file:
                log_file.write(message + "\n")
                log_file.flush()
        
        start_time = time.time()
        
        with open(data_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Statistics tracking
        stats = {
            'nodes_processed': 0,
            'nodes_created': 0,
            'nodes_updated': 0,
            'relationships_processed': 0,
            'relationships_created': 0,
            'relationships_failed': 0,
            'labels_count': Counter(),
            'relationship_types_count': Counter()
        }
        
        # Track nodes by their internal ID and custom ID properties
        node_ids = {}  # Maps node keys to Neo4j internal IDs
        id_mapping = {}  # Maps custom ID properties to Neo4j internal IDs
        
        log("\n" + "="*80)
        log("📦 IMPORTING NODES")
        log("="*80)
        if domain:
            log(f"🏷️  Domain tag: '{domain}'")
        log(f"📊 Total nodes to process: {len(data.get('nodes', []))}")
        log("")
        
        # Import nodes
        with self.driver.session() as session:
            for i, node in enumerate(data.get("nodes", []), 1):
                label = node.get("label")
                properties = node.get("properties", {})
                
                # ADD DOMAIN TAG TO ALL NODES
                if domain:
                    properties["domain"] = domain
                
                # Store the node reference
                node_key = f"{label}_{i}"
                
                # Create/update the node in Neo4j
                internal_id, custom_id = session.execute_write(
                    self.create_node, label, properties
                )
                
                node_ids[node_key] = internal_id
                
                # If the node has an ID property, store it for relationship mapping
                if "id" in properties:
                    id_mapping[properties["id"]] = internal_id
                
                # Update statistics
                stats['nodes_processed'] += 1
                stats['labels_count'][label] += 1
                
                # Progress indicator (every 50 nodes)
                if i % 50 == 0 or i == len(data.get("nodes", [])):
                    log(f"  Progress: {i}/{len(data.get('nodes', []))} nodes processed ({i/len(data.get('nodes', []))*100:.1f}%)")
        
        log(f"\n✅ Node import complete: {stats['nodes_processed']} nodes processed")
        
        # Import relationships
        log("\n" + "="*80)
        log("🔗 IMPORTING RELATIONSHIPS")
        log("="*80)
        log(f"📊 Total relationships to process: {len(data.get('relationships', []))}")
        log("")
        
        with self.driver.session() as session:
            for i, rel in enumerate(data.get("relationships", []), 1):
                from_id = rel.get("from")
                to_id = rel.get("to")
                rel_type = rel.get("type")
                properties = rel.get("properties", {})
                
                # Find the Neo4j internal IDs for the from/to nodes
                from_internal_id = id_mapping.get(from_id)
                to_internal_id = None
                
                # For the target node, first try direct ID mapping
                if to_id in id_mapping:
                    to_internal_id = id_mapping[to_id]
                else:
                    # If not in ID mapping, try to find the node by property
                    for node in data.get("nodes", []):
                        props = node.get("properties", {})
                        # Check if this node has a matching name, descrizione, or other identifying property
                        if (props.get("nome") == to_id or 
                            props.get("descrizione") == to_id or
                            props.get("tipo") == to_id):
                            # Node found by property, now get its internal ID
                            node_label = node.get("label")
                            for key, internal_id in node_ids.items():
                                if key.startswith(node_label):
                                    to_internal_id = internal_id
                                    break
                            if to_internal_id:
                                break
                
                # If still not found, try to find it directly in the database
                if not from_internal_id:
                    from_internal_id = session.execute_read(self.find_node_by_id, from_id)
                
                if not to_internal_id:
                    to_internal_id = session.execute_read(self.find_node_by_id, to_id)
                
                if from_internal_id and to_internal_id:
                    session.execute_write(
                        self.create_relationship, 
                        from_internal_id, 
                        to_internal_id, 
                        rel_type, 
                        properties
                    )
                    stats['relationships_processed'] += 1
                    stats['relationships_created'] += 1
                    stats['relationship_types_count'][rel_type] += 1
                else:
                    stats['relationships_processed'] += 1
                    stats['relationships_failed'] += 1
                    if not from_internal_id:
                        log(f"  ⚠️  Warning: Could not find source node with ID: {from_id}")
                    if not to_internal_id:
                        log(f"  ⚠️  Warning: Could not find target node with ID: {to_id}")
                
                # Progress indicator (every 50 relationships)
                if i % 50 == 0 or i == len(data.get("relationships", [])):
                    log(f"  Progress: {i}/{len(data.get('relationships', []))} relationships processed ({i/len(data.get('relationships', []))*100:.1f}%)")
        
        log(f"\n✅ Relationship import complete: {stats['relationships_created']} created, {stats['relationships_failed']} failed")
        
        # Final statistics
        elapsed_time = time.time() - start_time
        
        log("\n" + "="*80)
        log("📊 IMPORT STATISTICS")
        log("="*80)
        log(f"\n⏱️  Total time: {elapsed_time:.2f} seconds")
        log(f"\n📦 NODES:")
        log(f"  • Total processed: {stats['nodes_processed']}")
        log(f"  • Unique labels: {len(stats['labels_count'])}")
        log(f"\n🔗 RELATIONSHIPS:")
        log(f"  • Total processed: {stats['relationships_processed']}")
        log(f"  • Successfully created: {stats['relationships_created']}")
        log(f"  • Failed: {stats['relationships_failed']}")
        log(f"  • Unique types: {len(stats['relationship_types_count'])}")
        
        log(f"\n🏷️  TOP 10 NODE LABELS:")
        for label, count in stats['labels_count'].most_common(10):
            log(f"  {count:4d}  {label}")
        
        log(f"\n🔗 TOP 10 RELATIONSHIP TYPES:")
        for rel_type, count in stats['relationship_types_count'].most_common(10):
            log(f"  {count:4d}  {rel_type}")
        
        log("\n" + "="*80)
        log("✅ DATA IMPORT COMPLETED!")
        log("="*80)

def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(
        description='Import data from JSON into Neo4j with optional domain tagging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clear database and import UDL data with domain tag
  python data_ingestion_neo4j.py --uri neo4j+s://xxx.databases.neo4j.io --user neo4j --password XXX --file concepts4_neo4j.json --domain udl --clear
  
  # Add Neuro data with domain tag (without clearing)
  python data_ingestion_neo4j.py --uri neo4j+s://xxx.databases.neo4j.io --user neo4j --password XXX --file kg_neuro_neo4j.json --domain neuro
  
  # Clear only specific domain and re-import
  python data_ingestion_neo4j.py --uri neo4j+s://xxx.databases.neo4j.io --user neo4j --password XXX --file concepts4_neo4j.json --domain udl --clear-domain udl
        """
    )
    parser.add_argument('--uri', default='bolt://localhost:7687', help='Neo4j URI (default: bolt://localhost:7687)')
    parser.add_argument('--user', default='neo4j', help='Neo4j username (default: neo4j)')
    parser.add_argument('--password', required=True, help='Neo4j password (required)')
    parser.add_argument('--file', default='concepts_neo4j.json', help='JSON file to import (default: concepts_neo4j.json)')
    parser.add_argument('--domain', help='Domain tag to add to all nodes (e.g., udl, neuro, game) - RECOMMENDED for multi-domain KGs')
    parser.add_argument('--clear', action='store_true', help='Clear entire database before import (WARNING: deletes all data)')
    parser.add_argument('--clear-domain', metavar='DOMAIN', help='Clear only specific domain before import (e.g., --clear-domain udl)')
    
    args = parser.parse_args()
    
    # Validation
    if args.clear_domain and not args.domain:
        parser.error("--clear-domain requires --domain to be specified")
    
    if args.clear and args.clear_domain:
        parser.error("Cannot use both --clear and --clear-domain (choose one)")
    
    importer = Neo4jImporter(args.uri, args.user, args.password)
    
    # Create log file with timestamp
    from datetime import datetime
    log_filename = f"ingestion_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_file = open(log_filename, 'w', encoding='utf-8')
    
    def log_both(message):
        """Print and write to log file"""
        print(message)
        log_file.write(message + "\n")
        log_file.flush()
    
    try:
        log_both("="*80)
        log_both("🚀 NEO4J DATA INGESTION")
        log_both("="*80)
        log_both(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_both(f"📂 Input file: {args.file}")
        log_both(f"🏷️  Domain: {args.domain if args.domain else 'None'}")
        log_both(f"🌐 Neo4j URI: {args.uri}")
        log_both("="*80)
        
        # Clear operations
        if args.clear:
            log_both("\n⚠️  WARNING: Clearing entire database...")
            importer.clear_database()
        elif args.clear_domain:
            log_both(f"\n⚠️  WARNING: Clearing domain '{args.clear_domain}'...")
            deleted = importer.clear_domain(args.clear_domain)
        
        # Import data
        log_both(f"\n📥 Starting data import...")
        if args.domain:
            log_both(f"🏷️  Domain tag: {args.domain}")
        else:
            log_both("⚠️  No domain tag specified - nodes will not have domain property")
        
        importer.import_data(args.file, domain=args.domain, log_file=log_file)
        
        log_both("\n" + "="*80)
        log_both("✅ IMPORT COMPLETED SUCCESSFULLY!")
        log_both("="*80)
        log_both(f"📄 Log saved to: {log_filename}")
        
    except Exception as e:
        log_both(f"\n❌ Error during import: {e}")
        log_both(f"📄 Error log saved to: {log_filename}")
        raise
    finally:
        importer.close()
        log_file.close()

if __name__ == "__main__":
    # Change working directory to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    main() 