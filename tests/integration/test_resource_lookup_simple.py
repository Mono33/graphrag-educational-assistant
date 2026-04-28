"""
Simple test script for ResourceLookup module (standalone)

Tests only the ResourceLookup without importing full media module.
"""

import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_json_structure():
    """Test that the JSON file is valid and has correct structure"""
    print("\n" + "="*60)
    print("TEST 1: JSON Structure Validation")
    print("="*60)
    
    json_path = Path("kg_neuro_resources.json")
    
    if not json_path.exists():
        print(f"ERROR: File not found: {json_path}")
        return False
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"[OK] JSON is valid")
        
        # Check metadata
        metadata = data.get('metadata', {})
        print(f"[OK] Metadata found:")
        print(f"     Domain: {metadata.get('domain')}")
        print(f"     Version: {metadata.get('version')}")
        print(f"     Total resources: {metadata.get('statistics', {}).get('total_resources')}")
        
        # Check resources
        resources = data.get('resources', [])
        print(f"[OK] Resources array: {len(resources)} items")
        
        # Check topic mappings
        topic_mappings = data.get('topic_mappings', {})
        print(f"[OK] Topic mappings: {len(topic_mappings)} topics")
        
        # Check by_type
        by_type = data.get('by_type', {})
        print(f"[OK] Type groupings: {list(by_type.keys())}")
        
        # Check by_audience
        by_audience = data.get('by_audience', {})
        print(f"[OK] Audience groupings: {list(by_audience.keys())}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON - {e}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_resource_fields():
    """Test that all resources have required fields"""
    print("\n" + "="*60)
    print("TEST 2: Resource Fields Validation")
    print("="*60)
    
    required_fields = ['id', 'title', 'url', 'type', 'license', 'copyright_safe']
    
    try:
        with open("kg_neuro_resources.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        resources = data.get('resources', [])
        errors = []
        
        for r in resources:
            for field in required_fields:
                if field not in r:
                    errors.append(f"Resource '{r.get('id', 'UNKNOWN')}' missing field: {field}")
        
        if errors:
            print(f"ERRORS found:")
            for e in errors[:10]:
                print(f"  - {e}")
            return False
        else:
            print(f"[OK] All {len(resources)} resources have required fields")
            return True
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_copyright_safety():
    """Test that all resources are marked as copyright-safe"""
    print("\n" + "="*60)
    print("TEST 3: Copyright Safety Check")
    print("="*60)
    
    try:
        with open("kg_neuro_resources.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        resources = data.get('resources', [])
        
        unsafe = [r for r in resources if not r.get('copyright_safe', False)]
        
        if unsafe:
            print(f"WARNING: {len(unsafe)} resources NOT marked as copyright-safe:")
            for r in unsafe:
                print(f"  - {r.get('title')} ({r.get('license')})")
            return False
        else:
            print(f"[OK] All {len(resources)} resources are copyright-safe")
            
            # Show license distribution
            license_counts = {}
            for r in resources:
                lic = r.get('license', 'unknown')
                license_counts[lic] = license_counts.get(lic, 0) + 1
            
            print(f"\nLicense distribution:")
            for lic, count in sorted(license_counts.items()):
                print(f"  - {lic}: {count}")
            
            return True
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_topic_mappings():
    """Test that topic mappings reference valid resource IDs"""
    print("\n" + "="*60)
    print("TEST 4: Topic Mappings Validation")
    print("="*60)
    
    try:
        with open("kg_neuro_resources.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        resources = data.get('resources', [])
        resource_ids = {r['id'] for r in resources}
        
        topic_mappings = data.get('topic_mappings', {})
        
        errors = []
        for topic, ids in topic_mappings.items():
            for rid in ids:
                if rid not in resource_ids:
                    errors.append(f"Topic '{topic}' references non-existent ID: {rid}")
        
        if errors:
            print(f"ERRORS found:")
            for e in errors[:10]:
                print(f"  - {e}")
            return False
        else:
            print(f"[OK] All topic mappings reference valid resource IDs")
            print(f"\nTopics covered:")
            for topic in sorted(topic_mappings.keys()):
                print(f"  - {topic}: {len(topic_mappings[topic])} resources")
            return True
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_resource_lookup_class():
    """Test the ResourceLookup class directly"""
    print("\n" + "="*60)
    print("TEST 5: ResourceLookup Class")
    print("="*60)
    
    try:
        # Import just the resource_lookup module directly
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Direct import to avoid external_apis dependency
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "resource_lookup", 
            Path("agent/media/resource_lookup.py")
        )
        resource_lookup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(resource_lookup)
        
        ResourceLookup = resource_lookup.ResourceLookup
        
        # Initialize
        lookup = ResourceLookup(domain="neuro")
        
        print(f"[OK] ResourceLookup initialized")
        print(f"     Loaded: {lookup.loaded}")
        print(f"     Resources: {len(lookup.resources)}")
        
        # Test find by topic
        collection = lookup.find_resources_for_topic("metacognition")
        print(f"[OK] find_resources_for_topic('metacognition'): {len(collection.resources)} results")
        
        # Test find by type
        textbooks = lookup.find_by_type("textbook")
        print(f"[OK] find_by_type('textbook'): {len(textbooks)} results")
        
        # Test find by audience
        k12_resources = lookup.find_by_audience("K-12")
        print(f"[OK] find_by_audience('K-12'): {len(k12_resources)} results")
        
        # Test stats
        stats = lookup.get_stats()
        print(f"[OK] get_stats():")
        print(f"     Total: {stats['total_resources']}")
        print(f"     By type: {stats['by_type']}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("EXPERT RESOURCES JSON VALIDATION")
    print("="*60)
    
    results = []
    
    results.append(("JSON Structure", test_json_structure()))
    results.append(("Resource Fields", test_resource_fields()))
    results.append(("Copyright Safety", test_copyright_safety()))
    results.append(("Topic Mappings", test_topic_mappings()))
    results.append(("ResourceLookup Class", test_resource_lookup_class()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("All tests passed! kg_neuro_resources.json is ready to use.")
    else:
        print("Some tests failed. Review the output above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
