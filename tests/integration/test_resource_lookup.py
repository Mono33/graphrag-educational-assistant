"""
Test script for ResourceLookup module

Verifies that expert-vetted resources are loaded and searchable.
"""

import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_resource_lookup():
    """Test the ResourceLookup class"""
    print("\n" + "="*60)
    print("TEST: ResourceLookup Class")
    print("="*60)
    
    try:
        from aix.agent.media import ResourceLookup, ResourceCollection, ExpertResource
        
        # Initialize lookup
        lookup = ResourceLookup(domain="neuro")
        
        print(f"\n✅ ResourceLookup initialized")
        print(f"   Loaded: {lookup.loaded}")
        
        # Get stats
        stats = lookup.get_stats()
        print(f"\n📊 Statistics:")
        for key, value in stats.items():
            if key != 'metadata':
                print(f"   {key}: {value}")
        
        # Test 1: Find resources by topic
        print("\n" + "-"*40)
        print("TEST 1: Find resources by topic")
        print("-"*40)
        
        test_topics = ["metacognition", "working_memory", "attention"]
        for topic in test_topics:
            resources = lookup.find_resources_for_topic(topic)
            print(f"\n🔍 Topic: '{topic}'")
            print(f"   Found: {len(resources.resources)} resources")
            for r in resources.resources[:2]:
                print(f"   - {r.title} ({r.type})")
        
        # Test 2: Find resources by type
        print("\n" + "-"*40)
        print("TEST 2: Find resources by type")
        print("-"*40)
        
        test_types = ["textbook", "interactive_simulation", "course"]
        for res_type in test_types:
            resources = lookup.find_by_type(res_type)
            print(f"\n📚 Type: '{res_type}'")
            print(f"   Found: {len(resources)} resources")
            for r in resources[:2]:
                print(f"   - {r.title}")
        
        # Test 3: Find resources by audience
        print("\n" + "-"*40)
        print("TEST 3: Find resources by audience")
        print("-"*40)
        
        test_audiences = ["K-12", "teacher_training", "university"]
        for audience in test_audiences:
            resources = lookup.find_by_audience(audience)
            print(f"\n👥 Audience: '{audience}'")
            print(f"   Found: {len(resources)} resources")
        
        # Test 4: Generate context string
        print("\n" + "-"*40)
        print("TEST 4: Context string generation")
        print("-"*40)
        
        resources = lookup.find_resources_for_topic("metacognition")
        context = resources.to_context_string(max_resources=3)
        print(f"\n📝 Context string preview (first 500 chars):")
        print(context[:500] + "..." if len(context) > 500 else context)
        
        # Test 5: Find for multiple concepts
        print("\n" + "-"*40)
        print("TEST 5: Find for multiple concepts")
        print("-"*40)
        
        concepts = ["Working Memory", "Selective Attention", "Metacognition"]
        combined = lookup.find_resources_for_concepts(concepts, max_per_concept=2)
        print(f"\n🔗 Combined resources for {concepts}:")
        print(f"   Total unique: {len(combined.resources)}")
        
        # Test 6: Recommended for purpose
        print("\n" + "-"*40)
        print("TEST 6: Recommended for purpose")
        print("-"*40)
        
        purposes = ["lesson_creation", "interactive_learning", "professional_development"]
        for purpose in purposes:
            resources = lookup.get_recommended_for(purpose)
            print(f"\n🎯 Purpose: '{purpose}'")
            print(f"   Found: {len(resources)} resources")
        
        print("\n" + "="*60)
        print("✅ All ResourceLookup tests passed!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_copyright_safety():
    """Verify all resources are copyright-safe"""
    print("\n" + "="*60)
    print("TEST: Copyright Safety Check")
    print("="*60)
    
    try:
        from aix.agent.media import ResourceLookup
        
        lookup = ResourceLookup(domain="neuro")
        all_resources = lookup.get_all_resources()
        
        unsafe_resources = [r for r in all_resources if not r.copyright_safe]
        
        if unsafe_resources:
            print(f"\n⚠️ Found {len(unsafe_resources)} resources marked as not copyright-safe:")
            for r in unsafe_resources:
                print(f"   - {r.title} ({r.license})")
        else:
            print(f"\n✅ All {len(all_resources)} resources are marked as copyright-safe!")
        
        # Show license distribution
        license_counts = {}
        for r in all_resources:
            license_counts[r.license] = license_counts.get(r.license, 0) + 1
        
        print(f"\n📜 License distribution:")
        for license_type, count in sorted(license_counts.items()):
            print(f"   {license_type}: {count}")
        
        return len(unsafe_resources) == 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("EXPERT RESOURCE LOOKUP TESTS")
    print("="*60)
    
    results = []
    
    # Test 1: ResourceLookup functionality
    results.append(("ResourceLookup Class", test_resource_lookup()))
    
    # Test 2: Copyright safety
    results.append(("Copyright Safety", test_copyright_safety()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("🎉 All tests passed! ResourceLookup is ready to use.")
    else:
        print("⚠️ Some tests failed. Review the output above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
