#!/usr/bin/env python3
"""
Quick test script to verify the zero-shot classification project works correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from utils.config import set_seed, get_device
        print("✅ Utils import successful")
    except Exception as e:
        print(f"❌ Utils import failed: {e}")
        return False
    
    try:
        from data.dataset import create_sample_dataset
        print("✅ Data import successful")
    except Exception as e:
        print(f"❌ Data import failed: {e}")
        return False
    
    try:
        from eval.metrics import ZeroShotEvaluator
        print("✅ Evaluation import successful")
    except Exception as e:
        print(f"❌ Evaluation import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality without loading heavy models."""
    print("\nTesting basic functionality...")
    
    try:
        from utils.config import set_seed, get_device
        
        # Test seed setting
        set_seed(42)
        print("✅ Seed setting works")
        
        # Test device detection
        device = get_device()
        print(f"✅ Device detection works: {device}")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Testing Zero-shot Image Classification Project")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed!")
        return False
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality tests failed!")
        return False
    
    print("\n🎉 All tests passed! Project is ready to use.")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run demo: python main.py --mode demo")
    print("3. Run evaluation: python main.py --mode evaluate")
    print("4. Run benchmark: python scripts/benchmark.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
