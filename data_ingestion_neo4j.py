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
        """Create a node with given label and properties"""
        props_string = ", ".join([f"{k}: ${k}" for k in properties.keys()])
        query = f"CREATE (n:{label} {{{props_string}}}) RETURN n.id as node_id, elementId(n) as element_id"
        result = tx.run(query, **properties)
        record = result.single()
        return record["element_id"], record.get("node_id")
    
    def create_relationship(self, tx, from_id, to_id, rel_type, properties=None):
        """Create a relationship between two nodes"""
        if properties:
            props_string = ", ".join([f"{k}: ${k}" for k in properties.keys()])
            query = f"""
            MATCH (a), (b) 
            WHERE elementId(a) = $from_id AND elementId(b) = $to_id 
            CREATE (a)-[r:{rel_type} {{{props_string}}}]->(b)
            """
            tx.run(query, from_id=from_id, to_id=to_id, **properties)
        else:
            query = f"""
            MATCH (a), (b) 
            WHERE elementId(a) = $from_id AND elementId(b) = $to_id 
            CREATE (a)-[:{rel_type}]->(b)
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
            print("Database cleared")
    
    def import_data(self, data_file):
        """Import data from a JSON file into Neo4j"""
        with open(data_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Track nodes by their internal ID and custom ID properties
        node_ids = {}  # Maps node keys to Neo4j internal IDs
        id_mapping = {}  # Maps custom ID properties to Neo4j internal IDs
        
        # Import nodes
        with self.driver.session() as session:
            print("Importing nodes...")
            for i, node in enumerate(data.get("nodes", [])):
                label = node.get("label")
                properties = node.get("properties", {})
                
                # Store the node reference
                node_key = f"{label}_{i}"
                
                # Create the node in Neo4j
                internal_id, custom_id = session.execute_write(
                    self.create_node, label, properties
                )
                
                node_ids[node_key] = internal_id
                
                # If the node has an ID property, store it for relationship mapping
                if "id" in properties:
                    id_mapping[properties["id"]] = internal_id
                
                print(f"Created {label} node with ID: {internal_id}")
        
        # Import relationships
        with self.driver.session() as session:
            print("\nImporting relationships...")
            for rel in data.get("relationships", []):
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
                    print(f"Created {rel_type} relationship from {from_id} to {to_id}")
                else:
                    if not from_internal_id:
                        print(f"Warning: Could not find source node with ID: {from_id}")
                    if not to_internal_id:
                        print(f"Warning: Could not find target node with ID: {to_id}")
                    print(f"Warning: Could not create relationship {rel_type} from {from_id} to {to_id}")
        
        print("\nData import completed!")

def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description='Import data from JSON into Neo4j')
    parser.add_argument('--uri', default='bolt://localhost:7687', help='Neo4j URI')
    parser.add_argument('--user', default='neo4j', help='Neo4j username')
    parser.add_argument('--password', required=True, help='Neo4j password')
    parser.add_argument('--clear', action='store_true', help='Clear the database before import')
    parser.add_argument('--file', default='concepts_neo4j.json', help='JSON file to import')
    
    args = parser.parse_args()
    
    importer = Neo4jImporter(args.uri, args.user, args.password)
    
    try:
        if args.clear:
            importer.clear_database()
        
        importer.import_data(args.file)
    finally:
        importer.close()

if __name__ == "__main__":
    # Change working directory to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    main() 