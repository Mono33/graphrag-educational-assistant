#!/usr/bin/env python3
"""
merge_kg_json.py - Merge a delta KG JSON into the root KG JSON

Merges nodes by properties.id and relationships by (from, to, type) tuple.
Creates a timestamped backup of the root JSON before overwriting.

Domain-agnostic: works for neuro, UDL, or any future domain.

Usage:
    python merge_kg_json.py --root kg_neuro_neo4j.json --delta NeuroData/kg_neuro_neo4j.json
    python merge_kg_json.py --root concepts4_neo4j.json --delta UDLData/kg_udl_delta.json
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def load_json(filepath: str) -> dict:
    with open(filepath, encoding='utf-8') as f:
        return json.load(f)


def merge_nodes(root_nodes: list, delta_nodes: list) -> tuple:
    """Merge nodes by properties.id. Delta overwrites root on collision."""
    node_index = {}
    for node in root_nodes:
        node_id = node.get('properties', {}).get('id', '')
        if node_id:
            node_index[node_id] = node

    added = 0
    updated = 0
    for node in delta_nodes:
        node_id = node.get('properties', {}).get('id', '')
        if not node_id:
            continue
        if node_id in node_index:
            node_index[node_id] = node
            updated += 1
        else:
            node_index[node_id] = node
            added += 1

    return list(node_index.values()), added, updated


def merge_relationships(root_rels: list, delta_rels: list) -> tuple:
    """Merge relationships by (from, to, type) tuple. Skip duplicates."""
    rel_keys = set()
    merged = []

    for rel in root_rels:
        key = (rel.get('from', ''), rel.get('to', ''), rel.get('type', ''))
        if key not in rel_keys:
            rel_keys.add(key)
            merged.append(rel)

    added = 0
    skipped = 0
    for rel in delta_rels:
        key = (rel.get('from', ''), rel.get('to', ''), rel.get('type', ''))
        if key not in rel_keys:
            rel_keys.add(key)
            merged.append(rel)
            added += 1
        else:
            skipped += 1

    return merged, added, skipped


LABEL_FIXES = {
    'Teachingpractices': 'TeachingPractices',
    'Executivefunctions': 'ExecutiveFunctions',
    'Cognitiveneuroscience': 'CognitiveNeuroscience',
    'Learningoutcomes': 'LearningOutcomes',
    'Cognitiveload': 'CognitiveLoad',
    'Emotionalregulation': 'EmotionalRegulation',
    'Sociallearning': 'SocialLearning',
    'Growthmindset': 'GrowthMindset',
    'Educationalmyths': 'EducationalMyths',
}


def fix_labels(nodes: list) -> int:
    """Fix known case-variant labels to match Neo4j (PascalCase)."""
    fixed = 0
    for node in nodes:
        old_label = node.get('label', '')
        if old_label in LABEL_FIXES:
            node['label'] = LABEL_FIXES[old_label]
            fixed += 1
    return fixed


def main():
    parser = argparse.ArgumentParser(
        description='Merge a delta KG JSON into the root KG JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge_kg_json.py --root kg_neuro_neo4j.json --delta NeuroData/kg_neuro_neo4j.json
  python merge_kg_json.py --root concepts4_neo4j.json --delta UDLData/kg_udl_delta.json
  python merge_kg_json.py --root kg_neuro_neo4j.json --delta NeuroData/kg_neuro_neo4j.json --no-backup
        """
    )
    parser.add_argument('--root', required=True, help='Path to the root KG JSON (will be overwritten)')
    parser.add_argument('--delta', required=True, help='Path to the delta KG JSON (new data to merge)')
    parser.add_argument('--no-backup', action='store_true', help='Skip creating a backup of the root JSON')

    args = parser.parse_args()

    print("=" * 60)
    print("  MERGE KG JSON (Local Delta Merge)")
    print("=" * 60)

    root_data = load_json(args.root)
    root_nodes = root_data.get('nodes', [])
    root_rels = root_data.get('relationships', [])
    print(f"\n  Root:  {args.root}")
    print(f"         {len(root_nodes)} nodes, {len(root_rels)} relationships")

    delta_data = load_json(args.delta)
    delta_nodes = delta_data.get('nodes', [])
    delta_rels = delta_data.get('relationships', [])
    print(f"  Delta: {args.delta}")
    print(f"         {len(delta_nodes)} nodes, {len(delta_rels)} relationships")

    if not args.no_backup:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        root_path = Path(args.root)
        backup_path = root_path.parent / f"{root_path.stem}_backup_{timestamp}{root_path.suffix}"
        shutil.copy2(args.root, backup_path)
        print(f"\n  Backup: {backup_path}")

    print("\n  Merging nodes...")
    merged_nodes, nodes_added, nodes_updated = merge_nodes(root_nodes, delta_nodes)
    print(f"    Added:   {nodes_added}")
    print(f"    Updated: {nodes_updated}")
    print(f"    Total:   {len(merged_nodes)}")

    print("\n  Merging relationships...")
    merged_rels, rels_added, rels_skipped = merge_relationships(root_rels, delta_rels)
    print(f"    Added:   {rels_added}")
    print(f"    Skipped: {rels_skipped} (duplicates)")
    print(f"    Total:   {len(merged_rels)}")

    labels_fixed = fix_labels(merged_nodes)
    if labels_fixed:
        print("\n  Label remediation:")
        print(f"    Fixed:   {labels_fixed} nodes with case-variant labels")

    merged_data = {
        "nodes": merged_nodes,
        "relationships": merged_rels
    }

    with open(args.root, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved: {args.root}")
    print(f"         {len(merged_nodes)} nodes, {len(merged_rels)} relationships")

    print("\n" + "=" * 60)
    print("  MERGE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
