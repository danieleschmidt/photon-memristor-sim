#!/usr/bin/env python3
"""
Simple deployment readiness verification
"""
import sys
import os
import subprocess

def test_build_and_package():
    """Test library builds and can be packaged"""
    print("Testing build and packaging...")
    
    try:
        # Test wheel build
        result = subprocess.run([
            "bash", "-c", 
            "source venv/bin/activate && maturin build --release"
        ], capture_output=True, text=True, cwd="/root/repo")
        
        if result.returncode == 0:
            print("✅ Wheel build successful")
            return True
        else:
            print(f"❌ Wheel build failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Build test failed: {e}")
        return False

def test_import_and_basic_usage():
    """Test import and basic functionality"""
    print("Testing import and basic usage...")
    
    try:
        result = subprocess.run([
            "bash", "-c", 
            """source venv/bin/activate && python -c "
import photon_memristor_sim as pms
array = pms.PyPhotonicArray('crossbar', 4, 4)
print('SUCCESS: Library functional')
"
"""
        ], capture_output=True, text=True, cwd="/root/repo")
        
        if result.returncode == 0 and "SUCCESS" in result.stdout:
            print("✅ Import and basic usage working")
            return True
        else:
            print(f"❌ Import test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_production_files():
    """Test production configuration files exist"""
    print("Testing production configuration...")
    
    required_files = [
        "README.md", 
        "DEPLOYMENT.md",
        "Cargo.toml",
        "pyproject.toml",
        "LICENSE"
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(f"/root/repo/{file}"):
            missing.append(file)
    
    if not missing:
        print("✅ All production files present")
        return True
    else:
        print(f"❌ Missing files: {missing}")
        return False

def test_documentation():
    """Test documentation completeness"""  
    print("Testing documentation...")
    
    docs_files = [
        "docs/ARCHITECTURE.md",
        "docs/DEVELOPMENT.md", 
        "docs/ROADMAP.md"
    ]
    
    present = sum(1 for f in docs_files if os.path.exists(f"/root/repo/{f}"))
    coverage = present / len(docs_files) * 100
    
    print(f"✅ Documentation coverage: {coverage:.0f}% ({present}/{len(docs_files)} files)")
    return coverage >= 75  # Require 75% documentation coverage

def main():
    """Run deployment readiness tests"""
    print("🚀 DEPLOYMENT READINESS VERIFICATION")
    print("=" * 50)
    
    tests = [
        test_production_files,
        test_documentation,
        test_build_and_package,
        test_import_and_basic_usage,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 DEPLOYMENT READY!")
        print("✅ All deployment requirements met")
        print("✅ Library builds successfully")
        print("✅ Basic functionality validated")
        print("✅ Production files complete")
        print("✅ Documentation adequate")
        return 0
    else:
        print("⚠️ Deployment issues found - address before release")
        return 1

if __name__ == "__main__":
    sys.exit(main())