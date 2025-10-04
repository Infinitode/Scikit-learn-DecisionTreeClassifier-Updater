import os
import tempfile
import numpy as np
import pickle
import subprocess
import sys
from pathlib import Path

def test_upgrade_process(model_name, input_dir):
    """Test the upgrade process (new environment)"""
    try:
        # Define paths for the upgrade process
        shell_path = os.path.join(input_dir, f"{model_name}_shell.pkl")
        state_path = os.path.join(input_dir, f"{model_name}_state.pkl")
        json_path = os.path.join(input_dir, f"{model_name}_metadata.json")
        output_path = os.path.join(input_dir, f"{model_name}_upgraded.pkl")
        
        # Check if all required files exist
        if not (os.path.exists(shell_path) and os.path.exists(state_path) and os.path.exists(json_path)):
            print(f"      Missing conversion artifacts for {model_name}")
            return False
        
        # Run upgrade command
        upgrade_cmd = [
            sys.executable, 'main.py', 'upgrade',
            '--shell_path', shell_path,
            '--upgraded_tree_state_path', state_path,
            '--model_json_path', json_path,
            '--output_path', output_path
        ]
        
        result_upgrade = subprocess.run(upgrade_cmd, capture_output=True, text=True)
        if result_upgrade.returncode != 0:
            print(f"      Upgrade failed: {result_upgrade.stderr}")
            return False
        
        # Verify the upgraded model can be loaded (try both pickle and joblib)
        try:
            # Try pickle first (since conversion was done with pickle format)
            with open(output_path, 'rb') as f:
                upgraded_model = pickle.load(f)
            print(f"      Successfully upgraded and loaded {model_name}")
        except:
            # If pickle fails, try joblib (common format)
            import joblib
            try:
                upgraded_model = joblib.load(output_path)
                print(f"      Successfully upgraded and loaded {model_name}")
            except Exception as load_error:
                print(f"      Failed to load upgraded model: {load_error}")
                return False
        
        return True
        
    except Exception as e:
        print(f"      Upgrade process failed: {str(e)}")
        return False

def test_tree_models_upgrade(input_dir):
    """Test tree-based models upgrade"""
    print("Testing Tree-Based Models Upgrade...")
    
    models = [
        'DecisionTreeClassifier',
        'DecisionTreeRegressor', 
        'RandomForestClassifier',
        'RandomForestRegressor'
    ]
    
    results = {}
    for model_name in models:
        try:
            print(f"  Testing {model_name}...")
            success = test_upgrade_process(model_name, input_dir)
            results[model_name] = success
        except Exception as e:
            print(f"    Failed {model_name}: {str(e)}")
            results[model_name] = False
    
    return results

def test_scalers_upgrade(input_dir):
    """Test scaler models upgrade"""
    print("Testing Scalers Upgrade...")
    
    models = [
        'StandardScaler',
        'MinMaxScaler'
    ]
    
    results = {}
    for model_name in models:
        try:
            print(f"  Testing {model_name}...")
            success = test_upgrade_process(model_name, input_dir)
            results[model_name] = success
        except Exception as e:
            print(f"    Failed {model_name}: {str(e)}")
            results[model_name] = False
    
    return results

def test_nearest_neighbors_upgrade(input_dir):
    """Test nearest neighbor models upgrade"""
    print("Testing Nearest Neighbor Models Upgrade...")
    
    models = [
        'KNeighborsClassifier',
        'KNeighborsRegressor',
        'NearestNeighbors'
    ]
    
    results = {}
    for model_name in models:
        try:
            print(f"  Testing {model_name}...")
            success = test_upgrade_process(model_name, input_dir)
            results[model_name] = success
        except Exception as e:
            print(f"    Failed {model_name}: {str(e)}")
            results[model_name] = False
    
    return results

def test_svm_models_upgrade(input_dir):
    """Test SVM models upgrade"""
    print("Testing SVM Models Upgrade...")
    
    models = [
        'SVC',
        'SVR'
    ]
    
    results = {}
    for model_name in models:
        try:
            print(f"  Testing {model_name}...")
            success = test_upgrade_process(model_name, input_dir)
            results[model_name] = success
        except Exception as e:
            print(f"    Failed {model_name}: {str(e)}")
            results[model_name] = False
    
    return results

def test_linear_models_upgrade(input_dir):
    """Test linear models upgrade"""
    print("Testing Linear Models Upgrade...")
    
    models = [
        'LogisticRegression',
        'Ridge',
        'Lasso'
    ]
    
    results = {}
    for model_name in models:
        try:
            print(f"  Testing {model_name}...")
            success = test_upgrade_process(model_name, input_dir)
            results[model_name] = success
        except Exception as e:
            print(f"    Failed {model_name}: {str(e)}")
            results[model_name] = False
    
    return results

def test_decomposition_models_upgrade(input_dir):
    """Test decomposition models upgrade"""
    print("Testing Decomposition Models Upgrade...")
    
    models = [
        'PCA',
        'IncrementalPCA',
        'KernelPCA'
    ]
    
    results = {}
    for model_name in models:
        try:
            print(f"  Testing {model_name}...")
            success = test_upgrade_process(model_name, input_dir)
            results[model_name] = success
        except Exception as e:
            print(f"    Failed {model_name}: {str(e)}")
            results[model_name] = False
    
    return results

def test_preprocessing_models_upgrade(input_dir):
    """Test preprocessing models upgrade"""
    print("Testing Preprocessing Models Upgrade...")
    
    models = [
        'PolynomialFeatures',
        'OneHotEncoder'
    ]
    
    results = {}
    for model_name in models:
        try:
            print(f"  Testing {model_name}...")
            success = test_upgrade_process(model_name, input_dir)
            results[model_name] = success
        except Exception as e:
            print(f"    Failed {model_name}: {str(e)}")
            results[model_name] = False
    
    return results

def test_pipeline_models_upgrade(input_dir):
    """Test pipeline models upgrade"""
    print("Testing Pipeline Models Upgrade...")
    
    models = [
        'Pipeline',
        'FeatureUnion',
        'ColumnTransformer'
    ]
    
    results = {}
    for model_name in models:
        try:
            print(f"  Testing {model_name}...")
            success = test_upgrade_process(model_name, input_dir)
            results[model_name] = success
        except Exception as e:
            print(f"    Failed {model_name}: {str(e)}")
            results[model_name] = False
    
    return results

def main():
    """Main upgrade test function"""
    print("Testing Scikit-learn Model Upgrade (New Environment)...\n")
    
    # Check if conversion artifacts directory exists
    input_dir = "conversion_artifacts"
    if not os.path.exists(input_dir):
        print(f"Error: '{input_dir}' directory not found. Please run 'python test_conversion.py' in the old environment first.")
        sys.exit(1)
    
    all_results = {}
    
    # Test all model categories
    test_functions = [
        ("Tree-Based Models", test_tree_models_upgrade),
        ("Scalers", test_scalers_upgrade),
        ("Nearest Neighbor Models", test_nearest_neighbors_upgrade),
        ("SVM Models", test_svm_models_upgrade),
        ("Linear Models", test_linear_models_upgrade),
        ("Decomposition Models", test_decomposition_models_upgrade),
        ("Preprocessing Models", test_preprocessing_models_upgrade),
        ("Pipeline Models", test_pipeline_models_upgrade)
    ]
    
    for category_name, test_func in test_functions:
        print(f"\n{'='*50}")
        print(f"Testing {category_name} Upgrade")
        print('='*50)
        
        try:
            category_results = test_func(input_dir)
            all_results[category_name] = category_results
        except Exception as e:
            print(f"Error in {category_name}: {str(e)}")
            all_results[category_name] = {}
    
    # Print summary
    print(f"\n{'='*60}")
    print("UPGRADE TEST SUMMARY")
    print('='*60)
    
    total_tests = 0
    passed_tests = 0
    
    for category, results in all_results.items():
        print(f"\n{category}:")
        for model, success in results.items():
            status = "PASS" if success else "FAIL"
            print(f"  {model}: {status}")
            total_tests += 1
            if success:
                passed_tests += 1
    
    print(f"\nTotal: {passed_tests}/{total_tests} upgrades passed")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%" if total_tests > 0 else "0%")
    
    if passed_tests == total_tests:
        print("\n🎉 All upgrades passed!")
        return 0
    else:
        print(f"\n❌ {total_tests - passed_tests} upgrades failed")
        return 1

if __name__ == "__main__":
    # Ensure we're in the correct directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    # Check if main.py exists
    if not Path('main.py').exists():
        print("Error: main.py not found in current directory. Please run this script from the Scikit-learn-Model-Updater repository root.")
        sys.exit(1)
    
    exit_code = main()
    sys.exit(exit_code)