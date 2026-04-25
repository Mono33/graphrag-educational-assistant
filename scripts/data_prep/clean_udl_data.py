#!/usr/bin/env python3
"""
clean_udl_data.py - Clean and standardize UDL data for KG ingestion

Phase 1: SCAN - Identify all issues across 8 columns
Phase 2: FIX  - Apply corrections and produce KG_UDL_FINAL.xlsx
"""

import pandas as pd
import sys
import re
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

INPUT_FILE = "UDLdata/KG_Data_Collection_UDL_03206.xlsx"
OUTPUT_FILE = "UDLdata/KG_UDL_FINAL.xlsx"


# ============================================================================
# KNOWN FIXES (populated after Phase 1 scan)
# ============================================================================

RELATIONSHIP_FIXES = {
    "ASSOCIETS_TO": "ASSOCIATES_TO",
}

CONCEPT_A_FIXES = {
    "Knowdlege Application Process": "Knowledge Application Process",
}

VALUE_A_CASE_FIXES = {
    "tactile sensitivity": "Tactile sensitivity",
}

VALUE_B_TYPO_FIXES = {
    "Rembember": "Remember",
}

VALUE_B_CASE_FIXES = {
    "cooperative Learning": "Cooperative Learning",
    "distraction": "Distraction",
    "Google docs": "Google Docs",
    "movement": "Movement",
    "project Based Learning": "Project Based Learning",
}

CONCEPT_B_CASE_FIXES = {
    "Analogical tool": "Analogical Tool",
    "Digital tool": "Digital Tool",
}

CATEGORY_B_TYPO_FIXES = {
    "Understaning Process": "Understanding Process",
}

CATEGORY_B_CASE_FIXES = {
    "Observable challenge": "Observable Challenge",
}


def scan_column(df, col_name):
    """Scan a single column for issues"""
    print(f"\n{'=' * 60}")
    print(f"COLUMN: {col_name}")
    print(f"  Unique: {df[col_name].nunique()}, Null: {df[col_name].isna().sum()}")
    print(f"{'=' * 60}")

    issues = []

    if df[col_name].dtype != "object":
        print("  (non-text column, skipping)")
        return issues

    vals = df[col_name].dropna()

    # 1. Whitespace issues
    ws_mask = vals.apply(lambda x: str(x) != str(x).strip())
    ws_count = ws_mask.sum()
    if ws_count > 0:
        ws_vals = vals[ws_mask].unique()
        print(f"\n  WHITESPACE ISSUES ({ws_count} rows):")
        for v in ws_vals[:10]:
            print(f"    repr={repr(v)}")
        issues.append(("whitespace", ws_count))

    # 2. Case-insensitive duplicates
    lower_map = defaultdict(list)
    for v in vals.unique():
        lower_map[str(v).strip().lower()].append(str(v))

    dupes = {k: v for k, v in lower_map.items() if len(v) > 1}
    if dupes:
        print(f"\n  CASE DUPLICATES ({len(dupes)} groups):")
        for key, variants in sorted(dupes.items())[:20]:
            counts = [int(df[col_name].eq(v).sum()) for v in variants]
            print(f"    {variants} counts={counts}")
        issues.append(("case_dupes", len(dupes)))

    # 3. Show all unique values if <= 50
    unique_vals = sorted(vals.unique(), key=str)
    if len(unique_vals) <= 50:
        print(f"\n  ALL VALUES ({len(unique_vals)}):")
        for v in unique_vals:
            count = int(df[col_name].eq(v).sum())
            print(f"    [{count:3d}x] {v}")
    else:
        print(f"\n  TOP 30 VALUES (of {len(unique_vals)}):")
        vc = df[col_name].value_counts().head(30)
        for v, count in vc.items():
            print(f"    [{count:3d}x] {v}")

    return issues


def phase1_scan(df):
    """Phase 1: Full scan of all 8 columns"""
    print("=" * 60)
    print("PHASE 1: SCANNING UDL DATA")
    print(f"File: {INPUT_FILE}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print("=" * 60)

    all_issues = {}
    for col in df.columns:
        issues = scan_column(df, col)
        if issues:
            all_issues[col] = issues

    # Summary
    print(f"\n\n{'=' * 60}")
    print("SCAN SUMMARY")
    print(f"{'=' * 60}")
    if all_issues:
        for col, issues in all_issues.items():
            for issue_type, count in issues:
                print(f"  {col}: {issue_type} ({count})")
    else:
        print("  No issues found!")

    return all_issues


def phase2_fix(df):
    """Phase 2: Apply all fixes and save cleaned file"""
    print(f"\n\n{'=' * 60}")
    print("PHASE 2: APPLYING FIXES")
    print(f"{'=' * 60}")

    total_fixes = 0

    # 1. Strip whitespace from ALL text columns
    for col in df.columns:
        if df[col].dtype == "object":
            before = df[col].copy()
            df[col] = df[col].apply(lambda x: str(x).strip() if pd.notna(x) else x)
            changed = (before != df[col]).sum()
            if changed > 0:
                print(f"\n  [{col}] Stripped whitespace: {changed} cells")
                total_fixes += changed

    # 2. Fix Relationship typos
    for old, new in RELATIONSHIP_FIXES.items():
        mask = df["Relationship"] == old
        count = mask.sum()
        if count > 0:
            df.loc[mask, "Relationship"] = new
            print(f"\n  [Relationship] {old} -> {new}: {count} rows")
            total_fixes += count

    # 3. Fix Concept A typos
    for old, new in CONCEPT_A_FIXES.items():
        mask = df["Concept A"] == old
        count = mask.sum()
        if count > 0:
            df.loc[mask, "Concept A"] = new
            print(f"\n  [Concept A] {old} -> {new}: {count} rows")
            total_fixes += count

    # 4. Fix Value A case inconsistencies
    for old, new in VALUE_A_CASE_FIXES.items():
        mask = df["Value A"] == old
        count = mask.sum()
        if count > 0:
            df.loc[mask, "Value A"] = new
            print(f"\n  [Value A] {old} -> {new}: {count} rows")
            total_fixes += count

    # 5. Fix Value B typos and case inconsistencies
    for old, new in VALUE_B_TYPO_FIXES.items():
        mask = df["Value B"] == old
        count = mask.sum()
        if count > 0:
            df.loc[mask, "Value B"] = new
            print(f"\n  [Value B] {old} -> {new}: {count} rows")
            total_fixes += count

    for old, new in VALUE_B_CASE_FIXES.items():
        mask = df["Value B"] == old
        count = mask.sum()
        if count > 0:
            df.loc[mask, "Value B"] = new
            print(f"\n  [Value B] {old} -> {new}: {count} rows")
            total_fixes += count

    # 6. Fix Concept B case inconsistencies
    for old, new in CONCEPT_B_CASE_FIXES.items():
        mask = df["Concept B"] == old
        count = mask.sum()
        if count > 0:
            df.loc[mask, "Concept B"] = new
            print(f"\n  [Concept B] {old} -> {new}: {count} rows")
            total_fixes += count

    # 7. Fix Category B typos and case inconsistencies
    for old, new in CATEGORY_B_TYPO_FIXES.items():
        mask = df["Category B"] == old
        count = mask.sum()
        if count > 0:
            df.loc[mask, "Category B"] = new
            print(f"\n  [Category B] {old} -> {new}: {count} rows")
            total_fixes += count

    for old, new in CATEGORY_B_CASE_FIXES.items():
        mask = df["Category B"] == old
        count = mask.sum()
        if count > 0:
            df.loc[mask, "Category B"] = new
            print(f"\n  [Category B] {old} -> {new}: {count} rows")
            total_fixes += count

    # 8. Fill missing descriptions
    null_desc = df["Description"].isna().sum()
    if null_desc > 0:
        for idx in df[df["Description"].isna()].index:
            row = df.loc[idx]
            auto_desc = (
                f"{row['Value A']} {row['Relationship'].lower().replace('_', ' ')} "
                f"{row['Value B']}."
            )
            df.loc[idx, "Description"] = auto_desc
            print(f"\n  [Description] Auto-filled row {idx}: {auto_desc[:80]}...")
        total_fixes += null_desc

    print(f"\n\nTotal fixes applied: {total_fixes}")

    # Save
    df.to_excel(OUTPUT_FILE, index=False, sheet_name="UDL_Cleaned")
    print(f"Saved: {OUTPUT_FILE} ({len(df)} rows)")

    return df


def phase3_verify(df):
    """Phase 3: Re-scan to verify all fixes applied"""
    print(f"\n\n{'=' * 60}")
    print("PHASE 3: VERIFICATION SCAN")
    print(f"{'=' * 60}")

    # Quick checks
    issues_found = False

    # Whitespace
    for col in df.columns:
        if df[col].dtype == "object":
            ws = df[col].apply(lambda x: str(x) != str(x).strip() if pd.notna(x) else False).sum()
            if ws > 0:
                print(f"  STILL HAS WHITESPACE: {col} ({ws} rows)")
                issues_found = True

    # Known typos and case fixes
    checks = [
        ("Relationship", RELATIONSHIP_FIXES),
        ("Concept A", CONCEPT_A_FIXES),
        ("Value A", VALUE_A_CASE_FIXES),
        ("Value B", VALUE_B_TYPO_FIXES),
        ("Value B", VALUE_B_CASE_FIXES),
        ("Concept B", CONCEPT_B_CASE_FIXES),
        ("Category B", CATEGORY_B_TYPO_FIXES),
        ("Category B", CATEGORY_B_CASE_FIXES),
    ]
    for col_name, fix_dict in checks:
        for old in fix_dict:
            if (df[col_name] == old).any():
                print(f"  STILL HAS ISSUE: {col_name} = {old}")
                issues_found = True

    # Null descriptions
    null_desc = df["Description"].isna().sum()
    if null_desc > 0:
        print(f"  STILL HAS NULL DESCRIPTIONS: {null_desc}")
        issues_found = True

    if not issues_found:
        print("  All checks passed!")

    # Final stats
    print(f"\n  Final stats:")
    print(f"    Rows: {len(df)}")
    print(f"    Category A unique: {df['Category A'].nunique()}")
    print(f"    Concept A unique: {df['Concept A'].nunique()}")
    print(f"    Value A unique: {df['Value A'].nunique()}")
    print(f"    Relationship types: {df['Relationship'].nunique()}")
    print(f"    Value B unique: {df['Value B'].nunique()}")
    print(f"    Concept B unique: {df['Concept B'].nunique()}")
    print(f"    Category B unique: {df['Category B'].nunique()}")
    print(f"    Description non-null: {df['Description'].notna().sum()}/{len(df)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean UDL data")
    parser.add_argument(
        "--scan-only", action="store_true", help="Only scan, don't fix"
    )
    args = parser.parse_args()

    df = pd.read_excel(INPUT_FILE)

    if args.scan_only:
        phase1_scan(df)
    else:
        phase1_scan(df)
        df = phase2_fix(df)
        phase3_verify(df)
