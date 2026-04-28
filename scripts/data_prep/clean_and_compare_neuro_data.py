#!/usr/bin/env python3
"""
clean_and_compare_neuro_data.py - Clean new Excel data and compare with existing KG

This script:
1. Reads the new Excel from domain experts
2. Cleans relationship type typos
3. Standardizes concept names
4. Compares with existing KG (kg_neuro_neo4j.json)
5. Generates a detailed comparison report
6. Outputs cleaned Excel ready for transformation

Usage:
    python clean_and_compare_neuro_data.py
"""

import pandas as pd
import json
import re
import sys
import io
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any

# Fix Windows console encoding for Unicode
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths - Using the CURATED cleaned Excel as input now.
# Phase 2 reorg: KG core JSON now lives under data/kg/{domain}/
NEW_EXCEL_PATH = "NeuroData/KG_NEURO_CLEANED.xlsx"  # Curated by user
EXISTING_KG_PATH = "data/kg/neuro/kg_neuro_neo4j.json"
OUTPUT_CLEANED_EXCEL = "NeuroData/KG_NEURO_FINAL.xlsx"  # Final version ready for transform
OUTPUT_REPORT = "NeuroData/KG_NEURO_FINAL_COMPARISON_REPORT.md"

# Category standardization (to match existing KG)
CATEGORY_CORRECTIONS = {
    "Teaching Techniques": "Teaching Practices",
}

# Relationship type corrections (typo -> correct)
RELATIONSHIP_CORRECTIONS = {
    # Typos found in new data
    "ENHANCESS": "ENHANCES",
    "ENHANCESD_BY": "ENHANCED_BY",
    "IS_ENHANCESD_BY": "IS_ENHANCED_BY",
    "IS_ENHANCESD_THROUGH": "IS_ENHANCED_THROUGH",
    "IS_IS_SUPPORTED_BY": "IS_SUPPORTED_BY",
    "IS_ISUPPORTED_BY": "IS_SUPPORTED_BY",
    "ENHANCESS_ENCODING_OF": "ENHANCES_ENCODING_OF",
    "FOSTERS_METACOGNITION_ON": "FOSTERS_METACOGNITION",
    "ENHANCES_COGNITIVE_LOAD_ON": "ENHANCES_COGNITIVE_LOAD",
    "IMPAIRS_COGNITIVE_LOAD_ON": "IMPAIRS_COGNITIVE_LOAD",
    "TRIGGERS_BOTTOM_UP": "TRIGGERS",
    "CANNOT GENERATE": "CANNOT_GENERATE",
    "FAILS TO GUIDE": "FAILS_TO_GUIDE",
    
    # Standardize verb forms
    "FACILITATE": "FACILITATES",
    "REINFORCE": "REINFORCES",
    "UNDERMINE": "UNDERMINES",
    "INCREASE": "INCREASES",
    "LEAD_TO": "LEADS_TO",
    "CONTRIBUTE_TO": "CONTRIBUTES_TO",
    "INTERFERE_WITH": "INTERFERES_WITH",
    "REQUIRE": "REQUIRES",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_concept_name(name: str) -> str:
    """Normalize concept name for comparison (lowercase, trimmed)"""
    if pd.isna(name):
        return ""
    return str(name).strip().lower()

def generate_concept_id(name: str) -> str:
    """Generate semantic ID from concept name"""
    id_base = normalize_concept_name(name)
    id_base = re.sub(r'[^\w\s-]', '', id_base)
    id_base = id_base.replace(' ', '_').replace('-', '_')
    id_base = re.sub(r'_+', '_', id_base)
    id_base = id_base.strip('_')
    return f"concept_{id_base}"

def load_existing_kg(filepath: str) -> Dict[str, Any]:
    """Load existing KG from JSON file"""
    print(f"📂 Loading existing KG from: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract concepts with their details
    concepts = {}
    for node in data.get('nodes', []):
        props = node.get('properties', {})
        name = props.get('name', '')
        name_lower = normalize_concept_name(name)
        
        if name_lower:
            concepts[name_lower] = {
                'id': props.get('id', ''),
                'name': name,
                'label': node.get('label', ''),
                'category': props.get('category', ''),
                'description': props.get('description', ''),
            }
    
    # Extract relationships
    relationships = set()
    for rel in data.get('relationships', []):
        from_id = rel.get('from', '')
        to_id = rel.get('to', '')
        rel_type = rel.get('type', '')
        relationships.add((from_id, to_id, rel_type))
    
    print(f"   ✅ Loaded {len(concepts)} concepts, {len(relationships)} relationships")
    
    return {
        'concepts': concepts,
        'relationships': relationships,
        'raw': data
    }

def load_new_excel(filepath: str) -> pd.DataFrame:
    """Load and initially clean the new Excel file"""
    print(f"📂 Loading new Excel from: {filepath}")
    
    df = pd.read_excel(filepath, sheet_name=0)
    
    # Keep only relevant columns
    expected_cols = ['Category A', 'Concept A', 'Value A', 'Relationship', 
                     'Value B', 'Concept B', 'Category B', 'Description']
    
    # Filter to only expected columns (ignore unnamed columns)
    df = df[[col for col in expected_cols if col in df.columns]]
    
    # Drop rows where essential columns are missing
    df = df.dropna(subset=['Value A', 'Value B', 'Relationship'])
    
    print(f"   ✅ Loaded {len(df)} rows (after removing empty rows)")
    
    return df

def clean_relationship_types(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clean relationship type typos and return correction counts"""
    corrections_made = defaultdict(int)
    
    def fix_relationship(rel_type):
        if pd.isna(rel_type):
            return rel_type
        
        rel_type = str(rel_type).strip().upper()
        
        # Apply corrections
        if rel_type in RELATIONSHIP_CORRECTIONS:
            corrections_made[f"{rel_type} → {RELATIONSHIP_CORRECTIONS[rel_type]}"] += 1
            return RELATIONSHIP_CORRECTIONS[rel_type]
        
        return rel_type
    
    df['Relationship'] = df['Relationship'].apply(fix_relationship)
    
    return df, dict(corrections_made)

def clean_concept_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize concept names (trim whitespace, etc.)"""
    
    def clean_name(name):
        if pd.isna(name):
            return name
        return str(name).strip()
    
    df['Value A'] = df['Value A'].apply(clean_name)
    df['Value B'] = df['Value B'].apply(clean_name)
    df['Concept A'] = df['Concept A'].apply(clean_name)
    df['Concept B'] = df['Concept B'].apply(clean_name)
    df['Category A'] = df['Category A'].apply(clean_name)
    df['Category B'] = df['Category B'].apply(clean_name)
    
    return df

def clean_categories(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Standardize category names to match existing KG"""
    corrections_made = defaultdict(int)
    
    def fix_category(cat):
        if pd.isna(cat):
            return cat
        
        cat_str = str(cat).strip()
        
        # Apply corrections
        if cat_str in CATEGORY_CORRECTIONS:
            corrections_made[f"{cat_str} → {CATEGORY_CORRECTIONS[cat_str]}"] += 1
            return CATEGORY_CORRECTIONS[cat_str]
        
        return cat_str
    
    df['Category A'] = df['Category A'].apply(fix_category)
    df['Category B'] = df['Category B'].apply(fix_category)
    
    return df, dict(corrections_made)

def extract_concepts_from_excel(df: pd.DataFrame) -> Dict[str, Dict]:
    """Extract unique concepts from cleaned Excel"""
    concepts = {}
    
    for _, row in df.iterrows():
        # Extract concept A
        name_a = row['Value A']
        name_a_lower = normalize_concept_name(name_a)
        if name_a_lower and name_a_lower not in concepts:
            concepts[name_a_lower] = {
                'name': name_a,
                'concept': row.get('Concept A', ''),
                'category': row.get('Category A', ''),
                'description': row.get('Description', ''),
            }
        
        # Extract concept B
        name_b = row['Value B']
        name_b_lower = normalize_concept_name(name_b)
        if name_b_lower and name_b_lower not in concepts:
            concepts[name_b_lower] = {
                'name': name_b,
                'concept': row.get('Concept B', ''),
                'category': row.get('Category B', ''),
                'description': row.get('Description', ''),
            }
    
    return concepts

def extract_relationships_from_excel(df: pd.DataFrame) -> Set[Tuple[str, str, str]]:
    """Extract relationships from cleaned Excel"""
    relationships = set()
    
    for _, row in df.iterrows():
        from_id = generate_concept_id(row['Value A'])
        to_id = generate_concept_id(row['Value B'])
        rel_type = str(row['Relationship']).strip().upper()
        relationships.add((from_id, to_id, rel_type))
    
    return relationships

def safe_str(val, max_len: int = None) -> str:
    """Safely convert value to string, handling NaN"""
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if max_len:
        return s[:max_len]
    return s

def compare_concepts(existing: Dict, new: Dict) -> Dict[str, Any]:
    """Compare existing and new concepts"""
    existing_names = set(existing.keys())
    new_names = set(new.keys())
    
    # Find differences
    truly_new = new_names - existing_names
    overlap = new_names & existing_names
    only_in_old = existing_names - new_names
    
    # Check for description changes in overlapping concepts
    description_changes = []
    for name in overlap:
        old_desc = safe_str(existing[name].get('description', ''), 100)
        new_desc = safe_str(new[name].get('description', ''), 100)
        
        # Only flag if descriptions are meaningfully different
        if old_desc and new_desc and old_desc != new_desc:
            description_changes.append({
                'concept': new[name]['name'],
                'old_description': old_desc,
                'new_description': new_desc,
            })
    
    # Check for category changes
    category_changes = []
    for name in overlap:
        old_cat = safe_str(existing[name].get('category', ''))
        new_cat = safe_str(new[name].get('category', ''))
        
        if old_cat and new_cat and old_cat != new_cat:
            category_changes.append({
                'concept': new[name]['name'],
                'old_category': old_cat,
                'new_category': new_cat,
            })
    
    return {
        'truly_new': truly_new,
        'overlap': overlap,
        'only_in_old': only_in_old,
        'description_changes': description_changes,
        'category_changes': category_changes,
    }

def compare_relationships(existing: Set, new: Set) -> Dict[str, Any]:
    """Compare existing and new relationships"""
    truly_new = new - existing
    overlap = new & existing
    only_in_old = existing - new
    
    return {
        'truly_new': truly_new,
        'overlap': overlap,
        'only_in_old': only_in_old,
    }

def generate_report(
    corrections_made: Dict[str, int],
    concept_comparison: Dict[str, Any],
    relationship_comparison: Dict[str, Any],
    new_concepts: Dict[str, Dict],
    existing_kg: Dict[str, Any],
    df_cleaned: pd.DataFrame
) -> str:
    """Generate detailed comparison report in Markdown"""
    
    report = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report.append(f"# KG Neuro Data Comparison Report")
    report.append(f"\n**Generated:** {timestamp}")
    report.append(f"\n---\n")
    
    # Summary
    report.append("## 📊 Executive Summary\n")
    report.append("| Metric | Count |")
    report.append("|--------|-------|")
    report.append(f"| Existing KG Concepts | {len(existing_kg['concepts'])} |")
    report.append(f"| New Excel Concepts | {len(new_concepts)} |")
    report.append(f"| **Truly NEW Concepts** | **{len(concept_comparison['truly_new'])}** |")
    report.append(f"| Overlapping Concepts | {len(concept_comparison['overlap'])} |")
    report.append(f"| Only in Old KG | {len(concept_comparison['only_in_old'])} |")
    report.append(f"| New Relationships | {len(relationship_comparison['truly_new'])} |")
    report.append(f"| Description Changes | {len(concept_comparison['description_changes'])} |")
    report.append(f"| Category Changes | {len(concept_comparison['category_changes'])} |")
    
    # Data Cleaning Section
    report.append("\n---\n")
    report.append("## 🧹 Data Cleaning Applied\n")
    
    if corrections_made:
        report.append("### Relationship Type Corrections\n")
        report.append("| Typo → Correction | Count |")
        report.append("|-------------------|-------|")
        for correction, count in sorted(corrections_made.items(), key=lambda x: -x[1]):
            report.append(f"| {correction} | {count} |")
        report.append(f"\n**Total corrections:** {sum(corrections_made.values())}")
    else:
        report.append("✅ No typos found - data is clean!")
    
    # New Concepts Section
    report.append("\n---\n")
    report.append("## 🆕 New Concepts (Not in Current KG)\n")
    report.append(f"**Total: {len(concept_comparison['truly_new'])} new concepts**\n")
    
    if concept_comparison['truly_new']:
        # Group by category/concept type
        new_by_type = defaultdict(list)
        for name in concept_comparison['truly_new']:
            concept_data = new_concepts[name]
            concept_type = concept_data.get('concept', 'Unknown')
            new_by_type[concept_type].append(concept_data['name'])
        
        for concept_type, names in sorted(new_by_type.items()):
            report.append(f"\n### {concept_type} ({len(names)})")
            for name in sorted(names)[:20]:  # Limit to 20 per category
                report.append(f"- {name}")
            if len(names) > 20:
                report.append(f"- ... and {len(names) - 20} more")
    
    # Description Changes Section
    report.append("\n---\n")
    report.append("## 📝 Description Changes (Existing Concepts)\n")
    
    if concept_comparison['description_changes']:
        report.append(f"**{len(concept_comparison['description_changes'])} concepts have different descriptions:**\n")
        
        for i, change in enumerate(concept_comparison['description_changes'][:30], 1):
            report.append(f"\n### {i}. {change['concept']}")
            report.append(f"- **OLD:** {change['old_description']}...")
            report.append(f"- **NEW:** {change['new_description']}...")
        
        if len(concept_comparison['description_changes']) > 30:
            report.append(f"\n... and {len(concept_comparison['description_changes']) - 30} more changes")
    else:
        report.append("✅ No description changes detected.")
    
    # Category Changes Section
    report.append("\n---\n")
    report.append("## 🏷️ Category Changes\n")
    
    if concept_comparison['category_changes']:
        report.append(f"**{len(concept_comparison['category_changes'])} concepts have different categories:**\n")
        report.append("| Concept | Old Category | New Category |")
        report.append("|---------|--------------|--------------|")
        for change in concept_comparison['category_changes'][:30]:
            report.append(f"| {change['concept']} | {change['old_category']} | {change['new_category']} |")
    else:
        report.append("✅ No category changes detected.")
    
    # New Relationships Section
    report.append("\n---\n")
    report.append("## 🔗 New Relationships\n")
    report.append(f"**Total: {len(relationship_comparison['truly_new'])} new relationships**\n")
    
    if relationship_comparison['truly_new']:
        # Count by type
        rel_type_counts = defaultdict(int)
        for from_id, to_id, rel_type in relationship_comparison['truly_new']:
            rel_type_counts[rel_type] += 1
        
        report.append("### By Relationship Type\n")
        report.append("| Relationship Type | Count |")
        report.append("|-------------------|-------|")
        for rel_type, count in sorted(rel_type_counts.items(), key=lambda x: -x[1])[:20]:
            report.append(f"| {rel_type} | {count} |")
    
    # Concepts Only in Old KG
    report.append("\n---\n")
    report.append("## ⚠️ Concepts Only in Old KG (Not in New Excel)\n")
    report.append(f"**Total: {len(concept_comparison['only_in_old'])} concepts**\n")
    report.append("*These concepts exist in Neo4j but are not in the new Excel. They will be preserved.*\n")
    
    if concept_comparison['only_in_old']:
        sample = list(concept_comparison['only_in_old'])[:30]
        for name in sorted(sample):
            old_data = existing_kg['concepts'].get(name, {})
            report.append(f"- {old_data.get('name', name)}")
        if len(concept_comparison['only_in_old']) > 30:
            report.append(f"- ... and {len(concept_comparison['only_in_old']) - 30} more")
    
    # Recommendations
    report.append("\n---\n")
    report.append("## 💡 Recommendations\n")
    report.append("1. **Review the cleaned Excel** (`KG_NEURO_CLEANED.xlsx`) before transformation")
    report.append("2. **Decide on description changes**: Keep old or use new descriptions?")
    report.append(f"3. **Run transformation**: `python transform_team_data.py --team neuro --input NeuroData/KG_NEURO_CLEANED.xlsx`")
    report.append("4. **Use incremental import** with `--domain neuro` (MERGE will handle duplicates)")
    
    return "\n".join(report)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    print("=" * 80)
    print("🔧 KG NEURO DATA CLEANING & COMPARISON")
    print("=" * 80)
    
    # Step 1: Load existing KG
    print("\n📦 STEP 1: Loading Existing KG")
    print("-" * 40)
    existing_kg = load_existing_kg(EXISTING_KG_PATH)
    
    # Step 2: Load new Excel
    print("\n📦 STEP 2: Loading New Excel")
    print("-" * 40)
    df_new = load_new_excel(NEW_EXCEL_PATH)
    
    # Step 3: Clean data
    print("\n🧹 STEP 3: Cleaning Data")
    print("-" * 40)
    
    # Clean concept names
    df_cleaned = clean_concept_names(df_new)
    print("   ✅ Concept names standardized")
    
    # Clean relationship types
    df_cleaned, rel_corrections = clean_relationship_types(df_cleaned)
    print(f"   ✅ Relationship types cleaned: {sum(rel_corrections.values())} corrections")
    
    if rel_corrections:
        print("\n   Relationship corrections:")
        for correction, count in sorted(rel_corrections.items(), key=lambda x: -x[1])[:10]:
            print(f"      • {correction}: {count}")
    
    # Clean categories (standardize to match existing KG)
    df_cleaned, cat_corrections = clean_categories(df_cleaned)
    print(f"   ✅ Categories standardized: {sum(cat_corrections.values())} corrections")
    
    if cat_corrections:
        print("\n   Category corrections:")
        for correction, count in sorted(cat_corrections.items(), key=lambda x: -x[1]):
            print(f"      • {correction}: {count}")
    
    # Combine all corrections for report
    corrections_made = {**rel_corrections, **cat_corrections}
    
    # Step 4: Extract and compare concepts
    print("\n📊 STEP 4: Comparing Concepts")
    print("-" * 40)
    
    new_concepts = extract_concepts_from_excel(df_cleaned)
    concept_comparison = compare_concepts(existing_kg['concepts'], new_concepts)
    
    print(f"   • Truly NEW concepts: {len(concept_comparison['truly_new'])}")
    print(f"   • Overlapping concepts: {len(concept_comparison['overlap'])}")
    print(f"   • Only in old KG: {len(concept_comparison['only_in_old'])}")
    print(f"   • Description changes: {len(concept_comparison['description_changes'])}")
    print(f"   • Category changes: {len(concept_comparison['category_changes'])}")
    
    # Step 5: Extract and compare relationships
    print("\n🔗 STEP 5: Comparing Relationships")
    print("-" * 40)
    
    new_relationships = extract_relationships_from_excel(df_cleaned)
    relationship_comparison = compare_relationships(existing_kg['relationships'], new_relationships)
    
    print(f"   • Truly NEW relationships: {len(relationship_comparison['truly_new'])}")
    print(f"   • Overlapping relationships: {len(relationship_comparison['overlap'])}")
    print(f"   • Only in old KG: {len(relationship_comparison['only_in_old'])}")
    
    # Step 6: Save cleaned Excel
    print("\n💾 STEP 6: Saving Cleaned Excel")
    print("-" * 40)
    
    df_cleaned.to_excel(OUTPUT_CLEANED_EXCEL, index=False, sheet_name='Cleaned')
    print(f"   ✅ Saved: {OUTPUT_CLEANED_EXCEL}")
    
    # Step 7: Generate and save report
    print("\n📝 STEP 7: Generating Report")
    print("-" * 40)
    
    report = generate_report(
        corrections_made,
        concept_comparison,
        relationship_comparison,
        new_concepts,
        existing_kg,
        df_cleaned
    )
    
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"   ✅ Saved: {OUTPUT_REPORT}")
    
    # Final Summary
    print("\n" + "=" * 80)
    print("✅ CLEANING & COMPARISON COMPLETE!")
    print("=" * 80)
    
    print(f"""
📁 Generated Files:
   1. {OUTPUT_CLEANED_EXCEL}
      └─ Final Excel ready for transformation (categories standardized)
   
   2. {OUTPUT_REPORT}
      └─ Detailed comparison report

📊 Key Findings:
   • {len(concept_comparison['truly_new'])} truly NEW concepts to add
   • {len(relationship_comparison['truly_new'])} new relationships to add
   • {len(concept_comparison['overlap'])} concepts already exist (will be skipped/merged)
   • {sum(corrections_made.values())} total corrections applied

🎯 Next Steps:
   1. Review the comparison report: {OUTPUT_REPORT}
   2. Review final Excel: {OUTPUT_CLEANED_EXCEL}
   3. Run transformation:
      python transform_team_data.py --team neuro --input {OUTPUT_CLEANED_EXCEL} --output-dir NeuroData
   4. Import to Neo4j (incremental - adds new, preserves old):
      python data_ingestion_neo4j.py --file NeuroData/kg_neuro_neo4j.json --domain neuro
    """)

if __name__ == "__main__":
    main()
