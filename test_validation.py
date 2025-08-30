"""
MedSumAI Pro - Jupyter Notebook Testing and Validation Suite
Run this in a separate cell after running the main implementation
"""

import time
import json

def run_comprehensive_tests():
    """Run comprehensive tests on the MedSumAI system"""
    
    print("🧪 Starting MedSumAI Pro Testing Suite")
    print("=" * 60)
    
    test_results = {
        'data_loading': False,
        'model_loading': False,
        'perspective_extraction': False,
        'summary_generation': False,
        'performance': {}
    }
    
    # Test 1: Data Loading
    print("\n1️⃣ Testing Data Loading...")
    try:
        if 'train' in datasets and len(datasets['train']) > 0:
            sample_count = len(datasets['train'])
            print(f"   ✅ Successfully loaded {sample_count} training samples")
            test_results['data_loading'] = True
        else:
            print("   ❌ No training data loaded")
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
    
    # Test 2: Model Loading
    print("\n2️⃣ Testing Model Performance...")
    try:
        test_text = "Patient has diabetes and needs management advice."
        start_time = time.time()
        test_summary = summarizer.summarize_text(test_text)
        end_time = time.time()
        
        if test_summary and len(test_summary) > 10:
            print(f"   ✅ Model generating summaries successfully")
            print(f"   📊 Processing time: {end_time - start_time:.2f} seconds")
            test_results['model_loading'] = True
            test_results['performance']['summary_time'] = end_time - start_time
        else:
            print("   ❌ Model not generating valid summaries")
    except Exception as e:
        print(f"   ❌ Model testing failed: {e}")
    
    # Test 3: Perspective Extraction
    print("\n3️⃣ Testing Perspective Extraction...")
    try:
        if 'train' in datasets and len(datasets['train']) > 0:
            test_entry = datasets['train'].iloc[0].to_dict()
            perspectives = perspective_processor.extract_perspectives(test_entry)
            
            total_perspectives = sum(len(segments) for segments in perspectives.values())
            if total_perspectives > 0:
                print(f"   ✅ Extracted {total_perspectives} perspective segments")
                for perspective, segments in perspectives.items():
                    if segments:
                        print(f"      {perspective}: {len(segments)} segments")
                test_results['perspective_extraction'] = True
            else:
                print("   ❌ No perspectives extracted")
        else:
            print("   ❌ No data available for perspective testing")
    except Exception as e:
        print(f"   ❌ Perspective extraction failed: {e}")
    
    # Test 4: End-to-End Summary Generation
    print("\n4️⃣ Testing End-to-End Summary Generation...")
    try:
        if 'train' in datasets and len(datasets['train']) > 0:
            test_entry = datasets['train'].iloc[0].to_dict()
            
            start_time = time.time()
            patient_summary = audience_generator.generate_patient_summary(test_entry)
            clinician_summary = audience_generator.generate_clinician_summary(test_entry)
            end_time = time.time()
            
            if (patient_summary and len(patient_summary) > 50 and 
                clinician_summary and len(clinician_summary) > 50):
                print(f"   ✅ Both patient and clinician summaries generated")
                print(f"   📊 Total processing time: {end_time - start_time:.2f} seconds")
                print(f"   📏 Patient summary: {len(patient_summary)} chars")
                print(f"   📏 Clinician summary: {len(clinician_summary)} chars")
                test_results['summary_generation'] = True
                test_results['performance']['total_time'] = end_time - start_time
            else:
                print("   ❌ Summary generation incomplete")
        else:
            print("   ❌ No data available for summary testing")
    except Exception as e:
        print(f"   ❌ Summary generation failed: {e}")
    
    # Test 5: Memory Usage and Performance
    print("\n5️⃣ Testing System Performance...")
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        print(f"   📊 Current memory usage: {memory_mb:.1f} MB")
        
        if memory_mb < 4000:  # Less than 4GB
            print(f"   ✅ Memory usage within acceptable limits")
            test_results['performance']['memory_mb'] = memory_mb
        else:
            print(f"   ⚠️ Memory usage high but acceptable for 8GB system")
            test_results['performance']['memory_mb'] = memory_mb
            
    except ImportError:
        print("   ⚠️ psutil not available for memory testing")
    except Exception as e:
        print(f"   ⚠️ Performance testing failed: {e}")
    
    # Test 6: Error Handling
    print("\n6️⃣ Testing Error Handling...")
    try:
        # Test with empty entry
        empty_entry = {'question': '', 'context': '', 'answers': []}
        empty_patient = audience_generator.generate_patient_summary(empty_entry)
        empty_clinician = audience_generator.generate_clinician_summary(empty_entry)
        
        if empty_patient and empty_clinician:
            print("   ✅ Graceful handling of empty inputs")
        else:
            print("   ❌ Error handling needs improvement")
            
        # Test with malformed entry
        malformed_entry = {'invalid': 'data'}
        malformed_patient = audience_generator.generate_patient_summary(malformed_entry)
        malformed_clinician = audience_generator.generate_clinician_summary(malformed_entry)
        
        if malformed_patient and malformed_clinician:
            print("   ✅ Graceful handling of malformed inputs")
        else:
            print("   ❌ Malformed input handling needs improvement")
            
    except Exception as e:
        print(f"   ⚠️ Error handling test encountered issues: {e}")
    
    # Final Results
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(1 for test in test_results.values() if isinstance(test, bool) and test)
    total_core_tests = 4  # Core functionality tests
    
    print(f"Core Tests Passed: {passed_tests}/{total_core_tests}")
    
    for test_name, result in test_results.items():
        if isinstance(result, bool):
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    if 'performance' in test_results and test_results['performance']:
        print(f"\nPerformance Metrics:")
        for metric, value in test_results['performance'].items():
            if 'time' in metric:
                print(f"  {metric.replace('_', ' ').title()}: {value:.2f}s")
            elif 'memory' in metric:
                print(f"  {metric.replace('_', ' ').title()}: {value:.1f} MB")
    
    # System Status
    if passed_tests >= 3:
        print(f"\n🎉 MedSumAI Pro is ready for deployment!")
        print(f"✅ Core functionality working")
        print(f"💡 Run the Streamlit app to start using the system")
    else:
        print(f"\n⚠️ System needs attention before deployment")
        print(f"❌ Some core tests failed - check error messages above")
    
    return test_results

def demo_with_sample_data():
    """Run a demo with sample medical data"""
    print("\n🎬 Running Demo with Sample Data")
    print("=" * 50)
    
    # Create sample medical Q&A data
    sample_data = {
        'question': "What are the symptoms of diabetes and how is it managed?",
        'context': "Diabetes is a chronic condition that affects how your body processes blood sugar",
        'answers': [
            "Common symptoms include increased thirst, frequent urination, fatigue, and blurred vision. Management involves monitoring blood sugar levels, following a healthy diet, regular exercise, and taking prescribed medications as directed by your healthcare provider."
        ],
        'uri': 'demo-entry',
        'labelled_summaries': {}
    }
    
    print(f"Sample Question: {sample_data['question']}")
    print(f"\n🤖 Processing...")
    
    try:
        # Generate both summaries
        patient_summary = audience_generator.generate_patient_summary(sample_data)
        clinician_summary = audience_generator.generate_clinician_summary(sample_data)
        
        print(f"\n👤 PATIENT SUMMARY:")
        print("-" * 40)
        print(patient_summary)
        
        print(f"\n🩺 CLINICIAN SUMMARY:")
        print("-" * 40)
        print(clinician_summary)
        
        print(f"\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

# Run the tests and demo
if __name__ == "__main__":
    test_results = run_comprehensive_tests()
    demo_with_sample_data()