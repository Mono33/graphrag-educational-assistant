#!/usr/bin/env python3
"""
transform_team_data.py - Universal transformer for team knowledge graph data

This script:
1. Reads corrected team Excel (8-column structure)
2. Generates intermediary 3-sheet Excel (original + nodes + relationships)
3. Generates final Neo4j-ready JSON

Usage:
    python transform_team_data.py --team neuro --input KG_NEURO_Origin_CORRECTED.xlsx
    python transform_team_data.py --team udl --input KG_UDL_Data.xlsx
    python transform_team_data.py --team game --input GBL_Knowledge.xlsx
"""

import pandas as pd
import json
import re
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

# ============================================================================
# LABEL TO CATEGORY MAPPING (for node categories)
# ============================================================================

LABEL_TO_CATEGORY = {
    # Cognitive Processes
    "Attention": "Attention Types",
    "Memory": "Memory Systems",
    "ExecutiveFunctions": "Executive Functions",
    "ProblemSolving": "Cognitive Skills",
    "CriticalThinking": "Higher-Order Cognition",
    
    # Affective Processes
    "Emotions": "Affective States",
    "Motivation": "Motivational Types",
    "Stress": "Affective States",
    "Mindset": "Belief Systems",
    
    # Metacognition
    "Metacognition": "Metacognitive Processes",
    
    # Learning Processes
    "LearningOutcomes": "Educational Results",
    "SocialLearning": "Learning Processes",
    "Creativity": "Creative Processes",
    
    # Support Systems
    "TeachingPractices": "Pedagogical Strategies",
    "EducationalSupport": "Support Systems",
    
    # UDL-specific
    "PedagogicalMethodology": "Teaching Methods",
    "StudentWithSpecialNeeds": "Special Education",
    "StudentCharacteristic": "Student Profiles",
    "LearningEnvironment": "Environmental Factors",
    
    # Neuroscience
    "Neuroplasticity": "Neuroscience Foundations",
    "SpecialEducationNeeds": "Special Education",
    "LiteracyNumeracy": "Academic Skills",
    "EducationalMyths": "Misconceptions",
    "CognitiveBiases": "Cognitive Biases",
    "PersonalGrowth": "Developmental Outcomes",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def sanitize_label(label: str) -> str:
    """Sanitize label to be Neo4j-compatible (no spaces, special chars)
    
    Neo4j labels cannot have spaces or special characters.
    Example: "Cognitive load" → "CognitiveLoad"
             "Affective–Motivational Modulation" → "AffectiveMotivationalModulation"
    
    Args:
        label: Raw label string (may have spaces, hyphens, special chars)
    
    Returns:
        Neo4j-compatible label (PascalCase, no spaces)
    """
    # Replace special separators with spaces first
    label = label.replace('–', ' ')  # En dash
    label = label.replace('—', ' ')  # Em dash
    label = label.replace('-', ' ')  # Hyphen
    label = label.replace('/', ' ')  # Slash
    label = label.replace('_', ' ')  # Underscore
    
    # Remove non-alphanumeric chars (except spaces)
    label = re.sub(r'[^\w\s]', '', label)
    
    # Split by spaces and capitalize each word (PascalCase)
    words = label.split()
    pascal_case = ''.join(word.capitalize() for word in words)
    
    return pascal_case


def generate_node_id(name: str) -> str:
    """Generate semantic ID from node name
    
    Example: "Selective Attention" → "concept_selective_attention"
    """
    # Convert to lowercase
    id_base = name.lower()
    
    # Replace special characters with underscores
    id_base = re.sub(r'[^\w\s-]', '', id_base)  # Remove special chars except spaces and hyphens
    id_base = id_base.replace(' ', '_').replace('-', '_')
    
    # Remove duplicate underscores
    id_base = re.sub(r'_+', '_', id_base)
    
    # Remove leading/trailing underscores
    id_base = id_base.strip('_')
    
    return f"concept_{id_base}"


def get_node_category(label: str, original_category: str = None) -> str:
    """Get appropriate category for a node based on its label
    
    Args:
        label: Consolidated label (e.g., "Attention", "Memory")
        original_category: Original category from Excel (fallback)
    
    Returns:
        Specific subcategory (e.g., "Attention Types", "Memory Systems")
    """
    # Try to get from mapping
    if label in LABEL_TO_CATEGORY:
        return LABEL_TO_CATEGORY[label]
    
    # Fallback to original category if provided
    if original_category and not pd.isna(original_category):
        return original_category
    
    # Default fallback
    return "Cognitive Processes"


def generate_node_description(name: str, label: str, category: str, context: List[str] = None) -> str:
    """Generate educational description for a node
    
    Args:
        name: Node name (e.g., "Selective Attention")
        label: Node label (e.g., "Attention")
        category: Node category (e.g., "Attention Types")
        context: List of relationship descriptions mentioning this node
    
    Returns:
        Educational description
    """
    # If we have context from relationships, use it
    if context and len(context) > 0:
        # Use the first description as base (they already have good educational content)
        return context[0]
    
    # Otherwise, generate based on label
    templates = {
        "Attention": f"An attentional process involving {name.lower()}, essential for selective information processing and cognitive focus.",
        "Memory": f"A memory system related to {name.lower()}, crucial for encoding, storing, and retrieving information in learning contexts.",
        "ExecutiveFunctions": f"An executive function related to {name.lower()}, supporting goal-directed behavior, self-regulation, and cognitive control.",
        "Motivation": f"A motivational factor related to {name.lower()}, influencing engagement, persistence, and learning outcomes.",
        "Emotions": f"An emotional process or state involving {name.lower()}, affecting cognitive performance and learning experiences.",
        "Stress": f"A stress-related factor involving {name.lower()}, impacting cognitive functioning and emotional well-being.",
        "Mindset": f"A mindset or belief system related to {name.lower()}, shaping learning approaches and academic outcomes.",
        "Metacognition": f"A metacognitive process related to {name.lower()}, involving awareness and regulation of one's own thinking.",
        "LearningOutcomes": f"A learning outcome or result related to {name.lower()}, reflecting educational effectiveness and achievement.",
        "Creativity": f"A creative process or outcome related to {name.lower()}, supporting innovation, problem-solving, and divergent thinking.",
        "CriticalThinking": f"A critical thinking skill related to {name.lower()}, enabling analysis, evaluation, and reasoned judgment.",
        "SocialLearning": f"A social learning process related to {name.lower()}, involving interaction, communication, and collaborative knowledge construction.",
        "TeachingPractices": f"A teaching practice or instructional approach related to {name.lower()}, supporting effective pedagogy and student learning.",
        "EducationalSupport": f"An educational support system related to {name.lower()}, facilitating learning and addressing diverse student needs.",
        "PedagogicalMethodology": f"A pedagogical methodology related to {name.lower()}, providing systematic approaches to teaching and learning.",
        "StudentWithSpecialNeeds": f"A special education consideration related to {name.lower()}, addressing specific learning needs and accommodations.",
    }
    
    # Get template or use default
    template = templates.get(label, f"A {category.lower()} concept related to {name.lower()}, relevant to neuroscience of learning and educational practice.")
    
    return template


def validate_nodes_and_relationships(nodes: List[Dict], relationships: List[Dict]) -> Tuple[bool, List[str]]:
    """Validate that all relationships reference existing nodes
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Build set of all node IDs
    node_ids = {node['ID'] for node in nodes}
    
    # Check each relationship
    for idx, rel in enumerate(relationships, 1):
        from_id = rel['From_ID']
        to_id = rel['To_ID']
        
        if from_id not in node_ids:
            errors.append(f"Relationship {idx}: From_ID '{from_id}' not found in nodes")
        
        if to_id not in node_ids:
            errors.append(f"Relationship {idx}: To_ID '{to_id}' not found in nodes")
    
    return len(errors) == 0, errors


def generate_statistics(nodes: List[Dict], relationships: List[Dict]) -> Dict:
    """Generate statistics about the knowledge graph
    
    Returns:
        Dictionary with statistics
    """
    # Count nodes by label
    label_counts = defaultdict(int)
    for node in nodes:
        label_counts[node['Label']] += 1
    
    # Count relationships by type
    rel_type_counts = defaultdict(int)
    for rel in relationships:
        rel_type_counts[rel['Relationship_Type']] += 1
    
    # Calculate average nodes per label
    avg_nodes_per_label = len(nodes) / len(label_counts) if label_counts else 0
    
    return {
        'total_nodes': len(nodes),
        'total_relationships': len(relationships),
        'unique_labels': len(label_counts),
        'label_distribution': dict(sorted(label_counts.items(), key=lambda x: x[1], reverse=True)),
        'relationship_types': len(rel_type_counts),
        'relationship_distribution': dict(sorted(rel_type_counts.items(), key=lambda x: x[1], reverse=True)),
        'avg_nodes_per_label': round(avg_nodes_per_label, 1)
    }


# ============================================================================
# MAIN TRANSFORMATION FUNCTIONS
# ============================================================================

def extract_nodes_from_excel(df: pd.DataFrame) -> List[Dict]:
    """Extract unique nodes from 8-column Excel structure
    
    Args:
        df: DataFrame with columns: Category A, Concept A, Value A, Relationship, 
            Value B, Concept B, Category B, Description
    
    Returns:
        List of unique nodes with deduplication
    """
    print(f"\n📦 Extracting nodes...")
    
    seen_nodes = {}  # Track by ID to avoid duplicates
    node_contexts = defaultdict(list)  # Track descriptions mentioning each node
    
    for idx, row in df.iterrows():
        # Collect context (relationship descriptions) for each node
        description = row.get('Description', '')
        if not pd.isna(description):
            value_a_id = generate_node_id(row['Value A'])
            value_b_id = generate_node_id(row['Value B'])
            node_contexts[value_a_id].append(description)
            node_contexts[value_b_id].append(description)
        
        # Extract Node A
        node_a_id = generate_node_id(row['Value A'])
        if node_a_id not in seen_nodes:
            label_a = sanitize_label(row['Concept A'])  # ✅ Sanitize label
            category_a = get_node_category(label_a, row.get('Category A'))
            
            seen_nodes[node_a_id] = {
                'Label': label_a,
                'ID': node_a_id,
                'Name': row['Value A'],
                'Category': category_a,
                'Description': ''  # Will be filled later with context
            }
        
        # Extract Node B
        node_b_id = generate_node_id(row['Value B'])
        if node_b_id not in seen_nodes:
            label_b = sanitize_label(row['Concept B'])  # ✅ Sanitize label
            category_b = get_node_category(label_b, row.get('Category B'))
            
            seen_nodes[node_b_id] = {
                'Label': label_b,
                'ID': node_b_id,
                'Name': row['Value B'],
                'Category': category_b,
                'Description': ''  # Will be filled later with context
            }
    
    # Generate descriptions using context
    print(f"   ✅ Extracted {len(seen_nodes)} unique nodes (before deduplication: {len(df) * 2})")
    
    for node_id, node in seen_nodes.items():
        context = node_contexts.get(node_id, [])
        node['Description'] = generate_node_description(
            node['Name'], 
            node['Label'], 
            node['Category'],
            context
        )
    
    return list(seen_nodes.values())


def extract_relationships_from_excel(df: pd.DataFrame) -> List[Dict]:
    """Extract relationships from 8-column Excel structure
    
    Args:
        df: DataFrame with columns: Category A, Concept A, Value A, Relationship, 
            Value B, Concept B, Category B, Description
    
    Returns:
        List of relationships (one per row)
    """
    print(f"\n🔗 Extracting relationships...")
    
    relationships = []
    
    for idx, row in df.iterrows():
        from_id = generate_node_id(row['Value A'])
        to_id = generate_node_id(row['Value B'])
        
        relationships.append({
            'From_ID': from_id,
            'To_ID': to_id,
            'Relationship_Type': row['Relationship'],
            'From_Name': row['Value A'],
            'To_Name': row['Value B'],
            'Description': row.get('Description', '') if not pd.isna(row.get('Description')) else f"{row['Value A']} {row['Relationship'].lower().replace('_', ' ')} {row['Value B']}"
        })
    
    print(f"   ✅ Extracted {len(relationships)} relationships")
    
    return relationships


def create_intermediary_excel(df_original: pd.DataFrame, nodes: List[Dict], 
                              relationships: List[Dict], team_name: str, 
                              output_path: str):
    """Create 3-sheet Excel with original data, nodes, and relationships
    
    Args:
        df_original: Original 8-column DataFrame
        nodes: List of node dictionaries
        relationships: List of relationship dictionaries
        team_name: Name of the team (e.g., "neuro", "udl")
        output_path: Path to save Excel file
    """
    print(f"\n📊 Creating intermediary Excel with 3 sheets...")
    
    # Create DataFrames
    df_nodes = pd.DataFrame(nodes)
    df_relationships = pd.DataFrame(relationships)
    
    # Write to Excel with 3 sheets
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Original data
        df_original.to_excel(writer, sheet_name='original_data', index=False)
        print(f"   ✅ Sheet 1: 'original_data' ({len(df_original)} rows)")
        
        # Sheet 2: Nodes template
        df_nodes.to_excel(writer, sheet_name=f'{team_name}_template_nodes', index=False)
        print(f"   ✅ Sheet 2: '{team_name}_template_nodes' ({len(df_nodes)} rows)")
        
        # Sheet 3: Relationships template
        df_relationships.to_excel(writer, sheet_name=f'{team_name}_template_relationships', index=False)
        print(f"   ✅ Sheet 3: '{team_name}_template_relationships' ({len(df_relationships)} rows)")
    
    print(f"\n💾 Saved: {output_path}")


def create_neo4j_json(nodes: List[Dict], relationships: List[Dict], 
                     team_name: str, output_path: str):
    """Create Neo4j-ready JSON file
    
    Args:
        nodes: List of node dictionaries
        relationships: List of relationship dictionaries
        team_name: Name of the team (for domain tagging)
        output_path: Path to save JSON file
    """
    print(f"\n🔧 Creating Neo4j JSON...")
    
    # Convert to Neo4j format
    json_data = {
        "nodes": [
            {
                "label": node["Label"],
                "properties": {
                    "id": node["ID"],
                    "name": node["Name"],
                    "category": node["Category"],
                    "description": node["Description"],
                    "domain": team_name  # Add domain tag for filtering
                }
            }
            for node in nodes
        ],
        "relationships": [
            {
                "from": rel["From_ID"],
                "to": rel["To_ID"],
                "type": rel["Relationship_Type"],
                "properties": {
                    "from_name": rel["From_Name"],
                    "to_name": rel["To_Name"],
                    "description": rel["Description"]
                }
            }
            for rel in relationships
        ]
    }
    
    # Write JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Nodes: {len(json_data['nodes'])}")
    print(f"   ✅ Relationships: {len(json_data['relationships'])}")
    print(f"\n💾 Saved: {output_path}")


def transform_team_data(team_name: str, input_file: str, output_dir: str = "."):
    """Main transformation function - universal for all teams
    
    Args:
        team_name: Name of the team (e.g., "neuro", "udl", "game")
        input_file: Path to corrected Excel file (8-column structure)
        output_dir: Directory to save output files
    """
    print("=" * 80)
    print(f"🚀 TRANSFORM TEAM DATA - {team_name.upper()}")
    print("=" * 80)
    
    # Read input Excel
    print(f"\n📂 Reading input file: {input_file}")
    df = pd.read_excel(input_file, sheet_name=0)
    print(f"   ✅ Loaded {len(df)} rows")
    
    # Validate input structure
    expected_columns = ['Category A', 'Concept A', 'Value A', 'Relationship', 
                       'Value B', 'Concept B', 'Category B', 'Description']
    
    if not all(col in df.columns for col in expected_columns):
        print(f"\n❌ ERROR: Input file must have 8 columns:")
        print(f"   Expected: {expected_columns}")
        print(f"   Found: {list(df.columns)}")
        return
    
    # Extract nodes and relationships
    nodes = extract_nodes_from_excel(df)
    relationships = extract_relationships_from_excel(df)
    
    # Validate
    print(f"\n🔍 Validating data...")
    is_valid, errors = validate_nodes_and_relationships(nodes, relationships)
    
    if not is_valid:
        print(f"   ❌ Validation failed:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"      • {error}")
        if len(errors) > 10:
            print(f"      ... and {len(errors) - 10} more errors")
        return
    else:
        print(f"   ✅ Validation passed - all relationships reference existing nodes")
    
    # Generate statistics
    stats = generate_statistics(nodes, relationships)
    
    print(f"\n📊 STATISTICS:")
    print(f"   • Total nodes: {stats['total_nodes']}")
    print(f"   • Total relationships: {stats['total_relationships']}")
    print(f"   • Unique labels: {stats['unique_labels']}")
    print(f"   • Avg nodes per label: {stats['avg_nodes_per_label']}")
    print(f"   • Relationship types: {stats['relationship_types']}")
    
    print(f"\n📋 Top 10 Labels:")
    for i, (label, count) in enumerate(list(stats['label_distribution'].items())[:10], 1):
        print(f"   {i:2d}. {label:<30} {count:>4} nodes")
    
    # Define output paths
    excel_output = f"{output_dir}/KG_{team_name.upper()}_Transformed.xlsx"
    json_output = f"{output_dir}/kg_{team_name.lower()}_neo4j.json"
    
    # Create intermediary Excel
    create_intermediary_excel(df, nodes, relationships, team_name.lower(), excel_output)
    
    # Create Neo4j JSON
    create_neo4j_json(nodes, relationships, team_name.lower(), json_output)
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ TRANSFORMATION COMPLETE!")
    print("=" * 80)
    
    print(f"""
📁 Generated Files:
   1. {excel_output}
      └─ 3 sheets: original_data + {team_name}_template_nodes + {team_name}_template_relationships
   
   2. {json_output}
      └─ Ready for Neo4j ingestion with domain='{team_name}'

🎯 Next Steps:
   1. Review the Excel file (especially the nodes and relationships sheets)
   2. Make any manual edits if needed
   3. Ingest JSON into Neo4j:
      python data_ingestion_neo4j.py {json_output} --domain {team_name}

💡 Quality Metrics:
   • Avg nodes/label: {stats['avg_nodes_per_label']} (target: >10) {'✅' if stats['avg_nodes_per_label'] > 10 else '⚠️'}
   • Total nodes: {stats['total_nodes']}
   • Total relationships: {stats['total_relationships']}
   • Node-to-relationship ratio: {stats['total_nodes']/stats['total_relationships']:.2f}
    """)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Universal transformer for team knowledge graph data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transform_team_data.py --team neuro --input KG_NEURO_Origin_CORRECTED.xlsx
  python transform_team_data.py --team udl --input UDL_Data.xlsx --output-dir ./output
  python transform_team_data.py --team game --input GBL_Knowledge.xlsx
        """
    )
    
    parser.add_argument('--team', required=True, 
                       help='Team name (e.g., neuro, udl, game)')
    parser.add_argument('--input', required=True,
                       help='Path to corrected Excel file (8-column structure)')
    parser.add_argument('--output-dir', default='.',
                       help='Directory to save output files (default: current directory)')
    
    args = parser.parse_args()
    
    # Run transformation
    transform_team_data(args.team, args.input, args.output_dir)


if __name__ == "__main__":
    main()

