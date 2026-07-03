#!/usr/bin/env python3
"""
verify_ingestion.py - Verify multi-domain knowledge graph ingestion
Run this after ingesting data to verify everything is correct
"""

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

class IngestionVerifier:
    """Verify multi-domain knowledge graph structure"""

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def verify_domains(self):
        """Check domain distribution"""
        print("\n📊 DOMAIN DISTRIBUTION")
        print("=" * 60)

        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE n.domain IS NOT NULL
                RETURN n.domain as domain, count(*) as node_count
                ORDER BY node_count DESC
            """)

            total_nodes = 0
            for record in result:
                domain = record["domain"]
                count = record["node_count"]
                total_nodes += count
                print(f"  {domain.upper():10s}: {count:5d} nodes")

            print(f"  {'TOTAL':10s}: {total_nodes:5d} nodes")

    def verify_no_missing_domains(self):
        """Check for nodes without domain tags"""
        print("\n⚠️  NODES WITHOUT DOMAIN TAG")
        print("=" * 60)

        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE n.domain IS NULL
                RETURN count(n) as missing_count
            """)

            missing = result.single()["missing_count"]
            if missing == 0:
                print("  ✅ All nodes have domain tags!")
            else:
                print(f"  ⚠️  WARNING: {missing} nodes are missing domain tags")

                # Show sample
                sample = session.run("""
                    MATCH (n)
                    WHERE n.domain IS NULL
                    RETURN labels(n) as labels, properties(n) as props
                    LIMIT 5
                """)

                print("\n  Sample nodes without domain:")
                for record in sample:
                    print(f"    - {record['labels']}: {record['props']}")

    def verify_label_distribution(self):
        """Check label distribution per domain"""
        print("\n🏷️  LABEL DISTRIBUTION BY DOMAIN")
        print("=" * 60)

        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE n.domain IS NOT NULL
                RETURN n.domain as domain, labels(n)[0] as label, count(*) as count
                ORDER BY domain, count DESC
            """)

            current_domain = None
            for record in result:
                domain = record["domain"]
                label = record["label"]
                count = record["count"]

                if domain != current_domain:
                    print(f"\n  Domain: {domain.upper()}")
                    current_domain = domain

                print(f"    - {label:40s}: {count:4d} nodes")

    def verify_relationships(self):
        """Check relationship distribution"""
        print("\n🔗 RELATIONSHIP DISTRIBUTION")
        print("=" * 60)

        with self.driver.session() as session:
            result = session.run("""
                MATCH (a)-[r]->(b)
                WHERE a.domain IS NOT NULL AND b.domain IS NOT NULL
                RETURN a.domain as from_domain, type(r) as rel_type,
                       b.domain as to_domain, count(*) as count
                ORDER BY from_domain, count DESC
            """)

            current_domain = None
            total_rels = 0
            for record in result:
                from_domain = record["from_domain"]
                rel_type = record["rel_type"]
                to_domain = record["to_domain"]
                count = record["count"]
                total_rels += count

                if from_domain != current_domain:
                    print(f"\n  From Domain: {from_domain.upper()}")
                    current_domain = from_domain

                same_domain = "✅" if from_domain == to_domain else "🔀"
                print(f"    {same_domain} {rel_type:30s} → {to_domain:10s}: {count:4d} rels")

            print(f"\n  TOTAL RELATIONSHIPS: {total_rels}")

    def verify_sample_queries(self):
        """Test sample domain-filtered queries"""
        print("\n🔍 SAMPLE DOMAIN-FILTERED QUERIES")
        print("=" * 60)

        # UDL sample query
        print("\n  UDL Query: Find students with special needs")
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:StudentWithSpecialNeeds {domain: "udl"})
                RETURN s.name as name
                LIMIT 5
            """)

            for record in result:
                print(f"    - {record['name']}")

        # Neuro sample query
        print("\n  Neuro Query: Find attention-related concepts")
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:Attention {domain: "neuro"})
                RETURN a.name as name, a.category as category
                LIMIT 5
            """)

            for record in result:
                print(f"    - {record['name']} ({record['category']})")

    def verify_indexes(self):
        """Check if domain indexes exist"""
        print("\n📇 DOMAIN INDEXES")
        print("=" * 60)

        with self.driver.session() as session:
            result = session.run("SHOW INDEXES")

            domain_indexes = []
            for record in result:
                index_name = record.get("name", "")
                if "domain" in index_name.lower():
                    domain_indexes.append(index_name)

            if domain_indexes:
                print(f"  ✅ Found {len(domain_indexes)} domain indexes:")
                for idx in domain_indexes:
                    print(f"    - {idx}")
            else:
                print("  ⚠️  No domain indexes found!")
                print("  💡 Run create_domain_indexes.cypher to create them")

    def run_full_verification(self):
        """Run all verification checks"""
        print("\n" + "=" * 60)
        print("  MULTI-DOMAIN KNOWLEDGE GRAPH VERIFICATION")
        print("=" * 60)

        try:
            self.verify_domains()
            self.verify_no_missing_domains()
            self.verify_label_distribution()
            self.verify_relationships()
            self.verify_sample_queries()
            self.verify_indexes()

            print("\n" + "=" * 60)
            print("  ✅ VERIFICATION COMPLETE")
            print("=" * 60)
            print()

        except Exception as e:
            print(f"\n❌ Verification failed: {e}")
            raise

def main():
    """Main function"""
    # Load Neo4j credentials from environment variables
    # Set these in your .env file or export them before running
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

    if not all([NEO4J_URI, NEO4J_PASSWORD]):
        raise ValueError(
            "Missing Neo4j credentials!\n"
            "Please set NEO4J_URI and NEO4J_PASSWORD in your .env file\n"
            "Example:\n"
            "  NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io\n"
            "  NEO4J_PASSWORD=your-password-here"
        )

    verifier = IngestionVerifier(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        verifier.run_full_verification()
    finally:
        verifier.close()

if __name__ == "__main__":
    main()

