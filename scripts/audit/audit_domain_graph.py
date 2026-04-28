#!/usr/bin/env python3
"""
audit_domain_graph.py - Comprehensive audit of knowledge graph domains in Neo4j

This script:
1. Connects to Neo4j and analyzes any specified domain (UDL, Neuro, etc.)
2. Generates detailed statistics and quality metrics
3. Identifies patterns, anomalies, and potential issues
4. Exports comprehensive audit report (JSON + human-readable)

Usage:
    python audit_domain_graph.py --domain neuro --uri <uri> --password <pass>
    python audit_domain_graph.py --domain udl --uri <uri> --password <pass>
"""

import json
from neo4j import GraphDatabase
from collections import defaultdict, Counter
from datetime import datetime
import argparse

class DomainGraphAuditor:
    """Comprehensive auditor for knowledge graph domains in Neo4j
    
    Can audit any domain (UDL, Neuro, GameBased, etc.)
    """
    
    def __init__(self, uri: str, user: str, password: str, domain: str = "neuro"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.domain = domain
        self.audit_results = {
            "audit_metadata": {},
            "overview": {},
            "labels": {},
            "relationships": {},
            "nodes": {},
            "quality_metrics": {},
            "patterns": {},
            "recommendations": []
        }
    
    def close(self):
        """Close Neo4j connection"""
        self.driver.close()
    
    def run_full_audit(self):
        """Execute comprehensive audit"""
        print("=" * 80)
        print(f"🔍 COMPREHENSIVE NEO4J GRAPH AUDIT - Domain: {self.domain}")
        print("=" * 80)
        
        # Metadata
        self.audit_results["audit_metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "domain": self.domain,
            "neo4j_uri": self.driver._pool.address[0]
        }
        
        # Run audit sections
        print("\n📊 Section 1: Overview Statistics...")
        self._audit_overview()
        
        print("\n🏷️  Section 2: Label Analysis...")
        self._audit_labels()
        
        print("\n🔗 Section 3: Relationship Analysis...")
        self._audit_relationships()
        
        print("\n📋 Section 4: Node Analysis...")
        self._audit_nodes()
        
        print("\n✅ Section 5: Quality Metrics...")
        self._audit_quality()
        
        print("\n🔍 Section 6: Pattern Detection...")
        self._audit_patterns()
        
        print("\n💡 Section 7: Recommendations...")
        self._generate_recommendations()
        
        print("\n✅ Audit complete!")
        
        return self.audit_results
    
    def _audit_overview(self):
        """Get high-level overview statistics"""
        with self.driver.session() as session:
            # Total nodes and relationships
            result = session.run(f"""
                MATCH (n {{domain: $domain}})
                RETURN count(n) as node_count
            """, domain=self.domain)
            node_count = result.single()["node_count"]
            
            result = session.run(f"""
                MATCH (a {{domain: $domain}})-[r]->(b {{domain: $domain}})
                RETURN count(r) as rel_count
            """, domain=self.domain)
            rel_count = result.single()["rel_count"]
            
            # Unique labels
            result = session.run(f"""
                MATCH (n {{domain: $domain}})
                RETURN DISTINCT labels(n)[0] as label
            """, domain=self.domain)
            unique_labels = [record["label"] for record in result]
            
            # Unique relationship types
            result = session.run(f"""
                MATCH (a {{domain: $domain}})-[r]->(b {{domain: $domain}})
                RETURN DISTINCT type(r) as rel_type
            """, domain=self.domain)
            unique_rel_types = [record["rel_type"] for record in result]
            
            self.audit_results["overview"] = {
                "total_nodes": node_count,
                "total_relationships": rel_count,
                "unique_labels": len(unique_labels),
                "unique_relationship_types": len(unique_rel_types),
                "avg_relationships_per_node": round(rel_count / node_count, 2) if node_count > 0 else 0,
                "labels_list": unique_labels,
                "relationship_types_list": unique_rel_types
            }
            
            print(f"   ✅ Total nodes: {node_count}")
            print(f"   ✅ Total relationships: {rel_count}")
            print(f"   ✅ Unique labels: {len(unique_labels)}")
            print(f"   ✅ Unique relationship types: {len(unique_rel_types)}")
    
    def _audit_labels(self):
        """Analyze label distribution and properties"""
        with self.driver.session() as session:
            label_stats = {}
            
            # Get all labels
            result = session.run(f"""
                MATCH (n {{domain: $domain}})
                RETURN DISTINCT labels(n)[0] as label
            """, domain=self.domain)
            labels = [record["label"] for record in result]
            
            for label in labels:
                # Count nodes per label
                result = session.run(f"""
                    MATCH (n:{label} {{domain: $domain}})
                    RETURN count(n) as count
                """, domain=self.domain)
                count = result.single()["count"]
                
                # Get properties
                result = session.run(f"""
                    MATCH (n:{label} {{domain: $domain}})
                    RETURN keys(n) as props
                    LIMIT 1
                """, domain=self.domain)
                record = result.single()
                props = record["props"] if record else []
                
                # Get sample nodes
                result = session.run(f"""
                    MATCH (n:{label} {{domain: $domain}})
                    RETURN n.name as name
                    LIMIT 5
                """, domain=self.domain)
                samples = [record["name"] for record in result if record["name"]]
                
                # Outgoing relationships
                result = session.run(f"""
                    MATCH (n:{label} {{domain: $domain}})-[r]->()
                    RETURN count(r) as out_degree
                """, domain=self.domain)
                out_degree = result.single()["out_degree"]
                
                # Incoming relationships
                result = session.run(f"""
                    MATCH ()-[r]->(n:{label} {{domain: $domain}})
                    RETURN count(r) as in_degree
                """, domain=self.domain)
                in_degree = result.single()["in_degree"]
                
                label_stats[label] = {
                    "node_count": count,
                    "properties": props,
                    "sample_nodes": samples,
                    "outgoing_relationships": out_degree,
                    "incoming_relationships": in_degree,
                    "avg_out_degree": round(out_degree / count, 2) if count > 0 else 0,
                    "avg_in_degree": round(in_degree / count, 2) if count > 0 else 0
                }
            
            # Sort by node count
            sorted_labels = sorted(label_stats.items(), key=lambda x: x[1]["node_count"], reverse=True)
            
            self.audit_results["labels"] = {
                "total_labels": len(labels),
                "label_statistics": dict(sorted_labels),
                "top_10_labels": [{"label": k, "count": v["node_count"]} for k, v in sorted_labels[:10]]
            }
            
            print(f"   ✅ Analyzed {len(labels)} labels")
            print(f"   📊 Top 5 labels:")
            for i, (label, stats) in enumerate(sorted_labels[:5], 1):
                print(f"      {i}. {label}: {stats['node_count']} nodes")
    
    def _audit_relationships(self):
        """Analyze relationship patterns"""
        with self.driver.session() as session:
            # Get all relationship types
            result = session.run(f"""
                MATCH (a {{domain: $domain}})-[r]->(b {{domain: $domain}})
                RETURN DISTINCT type(r) as rel_type
            """, domain=self.domain)
            rel_types = [record["rel_type"] for record in result]
            
            rel_stats = {}
            for rel_type in rel_types:
                # Count relationships
                result = session.run(f"""
                    MATCH (a {{domain: $domain}})-[r:{rel_type}]->(b {{domain: $domain}})
                    RETURN count(r) as count
                """, domain=self.domain)
                count = result.single()["count"]
                
                # Get source->target label patterns
                result = session.run(f"""
                    MATCH (a {{domain: $domain}})-[r:{rel_type}]->(b {{domain: $domain}})
                    RETURN labels(a)[0] as source_label, labels(b)[0] as target_label, count(*) as pattern_count
                    ORDER BY pattern_count DESC
                    LIMIT 5
                """, domain=self.domain)
                patterns = [
                    {
                        "source": record["source_label"],
                        "target": record["target_label"],
                        "count": record["pattern_count"]
                    }
                    for record in result
                ]
                
                rel_stats[rel_type] = {
                    "count": count,
                    "top_patterns": patterns
                }
            
            # Sort by count
            sorted_rels = sorted(rel_stats.items(), key=lambda x: x[1]["count"], reverse=True)
            
            self.audit_results["relationships"] = {
                "total_relationship_types": len(rel_types),
                "relationship_statistics": dict(sorted_rels),
                "top_10_relationships": [{"type": k, "count": v["count"]} for k, v in sorted_rels[:10]]
            }
            
            print(f"   ✅ Analyzed {len(rel_types)} relationship types")
            print(f"   📊 Top 5 relationships:")
            for i, (rel_type, stats) in enumerate(sorted_rels[:5], 1):
                print(f"      {i}. {rel_type}: {stats['count']} occurrences")
    
    def _audit_nodes(self):
        """Analyze individual node characteristics"""
        with self.driver.session() as session:
            # Nodes with most connections
            result = session.run(f"""
                MATCH (n {{domain: $domain}})
                OPTIONAL MATCH (n)-[r]-()
                WITH n, count(r) as degree
                ORDER BY degree DESC
                LIMIT 10
                RETURN labels(n)[0] as label, n.name as name, degree
            """, domain=self.domain)
            most_connected = [
                {
                    "label": record["label"],
                    "name": record["name"],
                    "degree": record["degree"]
                }
                for record in result
            ]
            
            # Isolated nodes (no connections)
            result = session.run(f"""
                MATCH (n {{domain: $domain}})
                WHERE NOT (n)-[]-()
                RETURN count(n) as isolated_count
            """, domain=self.domain)
            isolated_count = result.single()["isolated_count"]
            
            # Nodes with missing properties
            result = session.run(f"""
                MATCH (n {{domain: $domain}})
                WHERE n.name IS NULL OR n.name = '' OR n.description IS NULL OR n.description = ''
                RETURN labels(n)[0] as label, n.name as name, n.id as id
                LIMIT 20
            """, domain=self.domain)
            missing_props = [
                {
                    "label": record["label"],
                    "name": record["name"],
                    "id": record["id"]
                }
                for record in result
            ]
            
            self.audit_results["nodes"] = {
                "most_connected_nodes": most_connected,
                "isolated_nodes_count": isolated_count,
                "nodes_with_missing_properties": missing_props,
                "missing_properties_count": len(missing_props)
            }
            
            print(f"   ✅ Most connected node: {most_connected[0]['name'] if most_connected else 'None'} ({most_connected[0]['degree'] if most_connected else 0} connections)")
            print(f"   ⚠️  Isolated nodes: {isolated_count}")
            print(f"   ⚠️  Nodes with missing properties: {len(missing_props)}")
    
    def _audit_quality(self):
        """Assess data quality metrics"""
        with self.driver.session() as session:
            # Property completeness
            result = session.run(f"""
                MATCH (n {{domain: $domain}})
                RETURN 
                    count(n) as total,
                    sum(CASE WHEN n.name IS NOT NULL AND n.name <> '' THEN 1 ELSE 0 END) as has_name,
                    sum(CASE WHEN n.category IS NOT NULL AND n.category <> '' THEN 1 ELSE 0 END) as has_category,
                    sum(CASE WHEN n.description IS NOT NULL AND n.description <> '' THEN 1 ELSE 0 END) as has_description
            """, domain=self.domain)
            record = result.single()
            total = record["total"]
            
            completeness = {
                "name_completeness": round((record["has_name"] / total * 100), 2) if total > 0 else 0,
                "category_completeness": round((record["has_category"] / total * 100), 2) if total > 0 else 0,
                "description_completeness": round((record["has_description"] / total * 100), 2) if total > 0 else 0
            }
            
            # Graph connectivity
            result = session.run(f"""
                MATCH (n {{domain: $domain}})
                OPTIONAL MATCH (n)-[r]-()
                WITH n, count(r) as degree
                RETURN 
                    avg(degree) as avg_degree,
                    min(degree) as min_degree,
                    max(degree) as max_degree
            """, domain=self.domain)
            record = result.single()
            connectivity = {
                "avg_degree": round(record["avg_degree"], 2),
                "min_degree": record["min_degree"],
                "max_degree": record["max_degree"]
            }
            
            # Calculate overall quality score (0-100)
            quality_score = (
                completeness["name_completeness"] * 0.4 +
                completeness["description_completeness"] * 0.3 +
                completeness["category_completeness"] * 0.2 +
                min(connectivity["avg_degree"] / 5 * 100, 100) * 0.1  # Target: 5 connections per node
            )
            
            self.audit_results["quality_metrics"] = {
                "property_completeness": completeness,
                "graph_connectivity": connectivity,
                "overall_quality_score": round(quality_score, 2),
                "quality_grade": self._get_quality_grade(quality_score)
            }
            
            print(f"   ✅ Name completeness: {completeness['name_completeness']}%")
            print(f"   ✅ Description completeness: {completeness['description_completeness']}%")
            print(f"   ✅ Avg connections per node: {connectivity['avg_degree']}")
            print(f"   ⭐ Overall quality score: {round(quality_score, 2)}/100 ({self._get_quality_grade(quality_score)})")
    
    def _get_quality_grade(self, score):
        """Convert quality score to grade"""
        if score >= 90:
            return "A (Excellent)"
        elif score >= 80:
            return "B (Good)"
        elif score >= 70:
            return "C (Fair)"
        elif score >= 60:
            return "D (Poor)"
        else:
            return "F (Needs Improvement)"
    
    def _audit_patterns(self):
        """Detect interesting patterns in the graph"""
        with self.driver.session() as session:
            # Hub nodes (high out-degree)
            result = session.run(f"""
                MATCH (n {{domain: $domain}})-[r]->()
                WITH n, count(r) as out_degree
                WHERE out_degree > 10
                RETURN labels(n)[0] as label, n.name as name, out_degree
                ORDER BY out_degree DESC
                LIMIT 10
            """, domain=self.domain)
            hubs = [{"label": r["label"], "name": r["name"], "out_degree": r["out_degree"]} for r in result]
            
            # Authority nodes (high in-degree)
            result = session.run(f"""
                MATCH ()-[r]->(n {{domain: $domain}})
                WITH n, count(r) as in_degree
                WHERE in_degree > 10
                RETURN labels(n)[0] as label, n.name as name, in_degree
                ORDER BY in_degree DESC
                LIMIT 10
            """, domain=self.domain)
            authorities = [{"label": r["label"], "name": r["name"], "in_degree": r["in_degree"]} for r in result]
            
            # Common triads (A->B->C patterns)
            result = session.run(f"""
                MATCH (a {{domain: $domain}})-[r1]->(b {{domain: $domain}})-[r2]->(c {{domain: $domain}})
                RETURN labels(a)[0] as label_a, type(r1) as rel1, labels(b)[0] as label_b, type(r2) as rel2, labels(c)[0] as label_c, count(*) as count
                ORDER BY count DESC
                LIMIT 10
            """, domain=self.domain)
            triads = [
                {
                    "pattern": f"{r['label_a']}-[{r['rel1']}]->{r['label_b']}-[{r['rel2']}]->{r['label_c']}",
                    "count": r["count"]
                }
                for r in result
            ]
            
            self.audit_results["patterns"] = {
                "hub_nodes": hubs,
                "authority_nodes": authorities,
                "common_triads": triads
            }
            
            print(f"   ✅ Found {len(hubs)} hub nodes (high outgoing connections)")
            print(f"   ✅ Found {len(authorities)} authority nodes (high incoming connections)")
            print(f"   ✅ Found {len(triads)} common triad patterns")
    
    def _generate_recommendations(self):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Check isolated nodes
        if self.audit_results["nodes"]["isolated_nodes_count"] > 0:
            recommendations.append({
                "priority": "HIGH",
                "category": "Connectivity",
                "issue": f"{self.audit_results['nodes']['isolated_nodes_count']} isolated nodes found",
                "recommendation": "Review isolated nodes and add relationships to integrate them into the knowledge graph"
            })
        
        # Check property completeness
        desc_completeness = self.audit_results["quality_metrics"]["property_completeness"]["description_completeness"]
        if desc_completeness < 90:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Data Quality",
                "issue": f"Description completeness is {desc_completeness}%",
                "recommendation": "Add descriptions to nodes missing them for better context and understanding"
            })
        
        # Check connectivity
        avg_degree = self.audit_results["quality_metrics"]["graph_connectivity"]["avg_degree"]
        if avg_degree < 3:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Connectivity",
                "issue": f"Average node connectivity is {avg_degree} (target: 5+)",
                "recommendation": "Enrich the graph by adding more relationships between related concepts"
            })
        
        # Check label distribution
        label_count = self.audit_results["overview"]["unique_labels"]
        node_count = self.audit_results["overview"]["total_nodes"]
        if label_count > node_count / 2:
            recommendations.append({
                "priority": "LOW",
                "category": "Schema",
                "issue": f"{label_count} labels for {node_count} nodes (many labels with few nodes)",
                "recommendation": "Consider consolidating similar labels to improve graph organization"
            })
        
        # Always recommend documentation
        recommendations.append({
            "priority": "LOW",
            "category": "Documentation",
            "issue": "Knowledge graph structure",
            "recommendation": "Document the schema, label meanings, and relationship semantics for team reference"
        })
        
        self.audit_results["recommendations"] = recommendations
        
        print(f"   ✅ Generated {len(recommendations)} recommendations")
        for i, rec in enumerate(recommendations, 1):
            print(f"      {i}. [{rec['priority']}] {rec['category']}: {rec['recommendation'][:80]}...")
    
    def save_report(self, output_file="neuro_audit_report.json", markdown_file="neuro_audit_report.md"):
        """Save audit results to files"""
        # Save JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.audit_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved JSON report: {output_file}")
        
        # Save Markdown
        self._generate_markdown_report(markdown_file)
        print(f"💾 Saved Markdown report: {markdown_file}")
    
    def _generate_markdown_report(self, output_file):
        """Generate human-readable Markdown report"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 🧠 Neuro Knowledge Graph Audit Report\n\n")
            
            # Metadata
            meta = self.audit_results["audit_metadata"]
            f.write(f"**Audit Date:** {meta['timestamp']}\n\n")
            f.write(f"**Domain:** {meta['domain']}\n\n")
            
            # Executive Summary
            overview = self.audit_results["overview"]
            quality = self.audit_results["quality_metrics"]
            
            f.write("## 📊 Executive Summary\n\n")
            f.write(f"- **Total Nodes:** {overview['total_nodes']}\n")
            f.write(f"- **Total Relationships:** {overview['total_relationships']}\n")
            f.write(f"- **Unique Labels:** {overview['unique_labels']}\n")
            f.write(f"- **Unique Relationship Types:** {overview['unique_relationship_types']}\n")
            f.write(f"- **Overall Quality Score:** {quality['overall_quality_score']}/100 ({quality['quality_grade']})\n\n")
            
            # Top Labels
            f.write("## 🏷️ Top 10 Labels\n\n")
            f.write("| Rank | Label | Node Count |\n")
            f.write("|------|-------|------------|\n")
            for i, item in enumerate(self.audit_results["labels"]["top_10_labels"], 1):
                f.write(f"| {i} | {item['label']} | {item['count']} |\n")
            f.write("\n")
            
            # Top Relationships
            f.write("## 🔗 Top 10 Relationship Types\n\n")
            f.write("| Rank | Relationship Type | Count |\n")
            f.write("|------|-------------------|-------|\n")
            for i, item in enumerate(self.audit_results["relationships"]["top_10_relationships"], 1):
                f.write(f"| {i} | {item['type']} | {item['count']} |\n")
            f.write("\n")
            
            # Quality Metrics
            f.write("## ✅ Quality Metrics\n\n")
            completeness = quality["property_completeness"]
            f.write(f"- **Name Completeness:** {completeness['name_completeness']}%\n")
            f.write(f"- **Category Completeness:** {completeness['category_completeness']}%\n")
            f.write(f"- **Description Completeness:** {completeness['description_completeness']}%\n\n")
            
            connectivity = quality["graph_connectivity"]
            f.write(f"- **Average Connections/Node:** {connectivity['avg_degree']}\n")
            f.write(f"- **Min Connections:** {connectivity['min_degree']}\n")
            f.write(f"- **Max Connections:** {connectivity['max_degree']}\n\n")
            
            # Most Connected Nodes
            f.write("## 🌟 Most Connected Nodes\n\n")
            f.write("| Rank | Node Name | Label | Connections |\n")
            f.write("|------|-----------|-------|-------------|\n")
            for i, node in enumerate(self.audit_results["nodes"]["most_connected_nodes"][:10], 1):
                f.write(f"| {i} | {node['name']} | {node['label']} | {node['degree']} |\n")
            f.write("\n")
            
            # Recommendations
            f.write("## 💡 Recommendations\n\n")
            for i, rec in enumerate(self.audit_results["recommendations"], 1):
                f.write(f"### {i}. [{rec['priority']}] {rec['category']}\n\n")
                f.write(f"**Issue:** {rec['issue']}\n\n")
                f.write(f"**Recommendation:** {rec['recommendation']}\n\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Audit knowledge graph domain in Neo4j (UDL, Neuro, etc.)')
    parser.add_argument('--uri', required=True, help='Neo4j URI')
    parser.add_argument('--user', default='neo4j', help='Neo4j username')
    parser.add_argument('--password', required=True, help='Neo4j password')
    parser.add_argument('--domain', default='neuro', help='Domain to audit (neuro, udl, gamebased, etc.)')
    parser.add_argument('--output', help='Output JSON file (default: <domain>_audit_report.json)')
    parser.add_argument('--markdown', help='Output Markdown file (default: <domain>_audit_report.md)')
    
    args = parser.parse_args()
    
    # Auto-generate output filenames based on domain if not specified
    if not args.output:
        args.output = f"{args.domain}_audit_report.json"
    if not args.markdown:
        args.markdown = f"{args.domain}_audit_report.md"
    
    auditor = DomainGraphAuditor(args.uri, args.user, args.password, args.domain)
    
    try:
        # Run audit
        auditor.run_full_audit()
        
        # Save reports
        auditor.save_report(args.output, args.markdown)
        
        print("\n" + "=" * 80)
        print("✅ AUDIT COMPLETE!")
        print("=" * 80)
        print(f"\n📁 Reports saved:")
        print(f"   • {args.output} (JSON)")
        print(f"   • {args.markdown} (Markdown)")
        
    finally:
        auditor.close()


if __name__ == "__main__":
    main()

