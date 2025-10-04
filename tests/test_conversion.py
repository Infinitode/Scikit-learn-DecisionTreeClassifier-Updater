import os
import tempfile
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle
import subprocess
import sys
from pathlib import Path

def create_test_data(task_type='classification', n_samples=100):
    """Create test data for different tasks"""
    if task_type == 'classification':
        X, y = make_classification(n_samples=n_samples, n_features=4, n_classes=2, n_informative=2, random_state=42)
    elif task_type == 'regression':
        X, y = make_regression(n_samples=n_samples, n_features=4, random_state=42)
    elif task_type == 'neighbors':
        X, y = make_classification(n_samples=n_samples, n_features=4, n_classes=2, n_informative=2, random_state=42)
    else:  # preprocessing
        X, y = make_classification(n_samples=n_samples, n_features=4, n_classes=2, n_informative=2, random_state=42)
    
    return X, y

def test_conversion_process(model, model_name, original_model_path, output_dir):
    """Test the conversion process (old environment)"""
    try:
        # Create output paths
        shell_path = os.path.join(output_dir, f"{model_name}_shell.pkl")
        state_path = os.path.join(output_dir, f"{model_name}_state.pkl")
        json_path = os.path.join(output_dir, f"{model_name}_metadata.json")
        
        # Run convert command
        convert_cmd = [
            sys.executable, 'main.py', 'convert',
            '--model_path', original_model_path,
            '--shell_path', shell_path,
            '--upgraded_tree_state_path', state_path,
            '--model_json_path', json_path
        ]
        
        result_convert = subprocess.run(convert_cmd, capture_output=True, text=True)
        if result_convert.returncode != 0:
            print(f"      Convert failed: {result_convert.stderr}")
            return False
        
        # Verify output files exist
        if not (os.path.exists(shell_path) and os.path.exists(state_path) and os.path.exists(json_path)):
            print(f"      Convert failed: output files not created")
            return False
        
        print(f"      Successfully converted {model_name}")
        return True
        
    except Exception as e:
        print(f"      Conversion process failed: {str(e)}")
        return False

def test_tree_models_conversion(output_dir):
    """Test tree-based models conversion"""
    print("Testing Tree-Based Models Conversion...")
    X, y = create_test_data('classification')
    X_reg, y_reg = create_test_data('regression')
    
    models = {
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
        'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=10, random_state=42),
        'RandomForestRegressor': RandomForestRegressor(n_estimators=10, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        try:
            print(f"  Testing {name}...")
            # Fit the model
            if 'Regressor' in name:
                model.fit(X_reg, y_reg)
            else:
                model.fit(X, y)
            
            # Save model using pickle (to match what the converter expects)
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                pickle.dump(model, f)
                original_model_path = f.name
            
            # Test conversion process
            success = test_conversion_process(model, name, original_model_path, output_dir)
            results[name] = success
            
            # Cleanup
            os.unlink(original_model_path)
            
        except Exception as e:
            print(f"    Failed {name}: {str(e)}")
            results[name] = False
    
    return results

def test_scalers_conversion(output_dir):
    """Test scaler models conversion"""
    print("Testing Scalers Conversion...")
    X, y = create_test_data('classification')
    
    models = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler()
    }
    
    results = {}
    for name, model in models.items():
        try:
            print(f"  Testing {name}...")
            # Fit the model
            model.fit(X)
            
            # Save model using pickle
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                pickle.dump(model, f)
                original_model_path = f.name
            
            # Test conversion process
            success = test_conversion_process(model, name, original_model_path, output_dir)
            results[name] = success
            
            # Cleanup
            os.unlink(original_model_path)
            
        except Exception as e:
            print(f"    Failed {name}: {str(e)}")
            results[name] = False
    
    return results

def test_nearest_neighbors_conversion(output_dir):
    """Test nearest neighbor models conversion"""
    print("Testing Nearest Neighbor Models Conversion...")
    X, y = create_test_data('neighbors')
    
    models = {
        'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=3),
        'KNeighborsRegressor': KNeighborsRegressor(n_neighbors=3),
        'NearestNeighbors': NearestNeighbors(n_neighbors=3)
    }
    
    results = {}
    for name, model in models.items():
        try:
            print(f"  Testing {name}...")
            # Fit the model
            if 'Regressor' in name:
                X_reg, y_reg = create_test_data('regression')
                model.fit(X_reg, y_reg)
            elif 'Classifier' in name:
                model.fit(X, y)
            else:  # NearestNeighbors
                model.fit(X)
            
            # Save model using pickle
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                pickle.dump(model, f)
                original_model_path = f.name
            
            # Test conversion process
            success = test_conversion_process(model, name, original_model_path, output_dir)
            results[name] = success
            
            # Cleanup
            os.unlink(original_model_path)
            
        except Exception as e:
            print(f"    Failed {name}: {str(e)}")
            results[name] = False
    
    return results

def test_svm_models_conversion(output_dir):
    """Test SVM models conversion"""
    print("Testing SVM Models Conversion...")
    X, y = create_test_data('classification')
    X_reg, y_reg = create_test_data('regression')
    
    models = {
        'SVC': SVC(kernel='linear', random_state=42),
        'SVR': SVR(kernel='linear')
    }
    
    results = {}
    for name, model in models.items():
        try:
            print(f"  Testing {name}...")
            # Fit the model
            if 'SVR' in name:
                model.fit(X_reg, y_reg)
            else:
                model.fit(X, y)
            
            # Save model using pickle
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                pickle.dump(model, f)
                original_model_path = f.name
            
            # Test conversion process
            success = test_conversion_process(model, name, original_model_path, output_dir)
            results[name] = success
            
            # Cleanup
            os.unlink(original_model_path)
            
        except Exception as e:
            print(f"    Failed {name}: {str(e)}")
            results[name] = False
    
    return results

def test_linear_models_conversion(output_dir):
    """Test linear models conversion"""
    print("Testing Linear Models Conversion...")
    X, y = create_test_data('classification')
    X_reg, y_reg = create_test_data('regression')
    
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'Ridge': Ridge(),
        'Lasso': Lasso()
    }
    
    results = {}
    for name, model in models.items():
        try:
            print(f"  Testing {name}...")
            # Fit the model
            if name == 'LogisticRegression':
                model.fit(X, y)
            else:
                model.fit(X_reg, y_reg)
            
            # Save model using pickle
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                pickle.dump(model, f)
                original_model_path = f.name
            
            # Test conversion process
            success = test_conversion_process(model, name, original_model_path, output_dir)
            results[name] = success
            
            # Cleanup
            os.unlink(original_model_path)
            
        except Exception as e:
            print(f"    Failed {name}: {str(e)}")
            results[name] = False
    
    return results

def test_decomposition_models_conversion(output_dir):
    """Test decomposition models conversion"""
    print("Testing Decomposition Models Conversion...")
    X, y = create_test_data('classification')
    
    models = {
        'PCA': PCA(n_components=2),
        'IncrementalPCA': IncrementalPCA(n_components=2),
        'KernelPCA': KernelPCA(n_components=2)
    }
    
    results = {}
    for name, model in models.items():
        try:
            print(f"  Testing {name}...")
            # Fit the model
            model.fit(X)
            
            # Save model using pickle
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                pickle.dump(model, f)
                original_model_path = f.name
            
            # Test conversion process
            success = test_conversion_process(model, name, original_model_path, output_dir)
            results[name] = success
            
            # Cleanup
            os.unlink(original_model_path)
            
        except Exception as e:
            print(f"    Failed {name}: {str(e)}")
            results[name] = False
    
    return results

def test_preprocessing_models_conversion(output_dir):
    """Test preprocessing models conversion"""
    print("Testing Preprocessing Models Conversion...")
    X, y = create_test_data('classification')
    
    # Create mixed data for OneHotEncoder test
    X_mixed = np.column_stack([X, np.random.choice([0, 1, 2], size=(X.shape[0], 1))])
    
    models = {
        'PolynomialFeatures': PolynomialFeatures(degree=2),
        'OneHotEncoder': OneHotEncoder(sparse_output=False)
    }
    
    results = {}
    for name, model in models.items():
        try:
            print(f"  Testing {name}...")
            # Fit the model
            if name == 'OneHotEncoder':
                model.fit(X_mixed[:, -1].reshape(-1, 1))
            else:
                model.fit(X)
            
            # Save model using pickle
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                pickle.dump(model, f)
                original_model_path = f.name
            
            # Test conversion process
            success = test_conversion_process(model, name, original_model_path, output_dir)
            results[name] = success
            
            # Cleanup
            os.unlink(original_model_path)
            
        except Exception as e:
            print(f"    Failed {name}: {str(e)}")
            results[name] = False
    
    return results

def test_pipeline_models_conversion(output_dir):
    """Test pipeline models conversion"""
    print("Testing Pipeline Models Conversion...")
    X, y = create_test_data('classification')
    X_reg, y_reg = create_test_data('regression')
    
    # Sample data for ColumnTransformer test
    X_mixed = np.column_stack([X, np.random.choice([0, 1, 2], size=(X.shape[0], 1))])
    
    models = {
        'Pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ]),
        'FeatureUnion': FeatureUnion([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=2))
        ]),
        'ColumnTransformer': ColumnTransformer([
            ('scaler', StandardScaler(), slice(0, 4)),
            ('encoder', OneHotEncoder(sparse_output=False), [4])
        ])
    }
    
    results = {}
    for name, model in models.items():
        try:
            print(f"  Testing {name}...")
            # Fit the model
            if name == 'ColumnTransformer':
                model.fit(X_mixed)
            elif 'Regressor' in str(type(model.steps[-1][1] if hasattr(model, 'steps') else model.transformer_list[0][1])):
                model.fit(X_reg, y_reg)
            else:
                model.fit(X, y)
            
            # Save model using pickle
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                pickle.dump(model, f)
                original_model_path = f.name
            
            # Test conversion process
            success = test_conversion_process(model, name, original_model_path, output_dir)
            results[name] = success
            
            # Cleanup
            os.unlink(original_model_path)
            
        except Exception as e:
            print(f"    Failed {name}: {str(e)}")
            results[name] = False
    
    return results

def main():
    """Main conversion test function"""
    print("Testing Scikit-learn Model Conversion (Old Environment)...\n")
    
    # Create output directory for conversion artifacts
    output_dir = "conversion_artifacts"
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    # Test all model categories
    test_functions = [
        ("Tree-Based Models", test_tree_models_conversion),
        ("Scalers", test_scalers_conversion),
        ("Nearest Neighbor Models", test_nearest_neighbors_conversion),
        ("SVM Models", test_svm_models_conversion),
        ("Linear Models", test_linear_models_conversion),
        ("Decomposition Models", test_decomposition_models_conversion),
        ("Preprocessing Models", test_preprocessing_models_conversion),
        ("Pipeline Models", test_pipeline_models_conversion)
    ]
    
    for category_name, test_func in test_functions:
        print(f"\n{'='*50}")
        print(f"Testing {category_name} Conversion")
        print('='*50)
        
        try:
            category_results = test_func(output_dir)
            all_results[category_name] = category_results
        except Exception as e:
            print(f"Error in {category_name}: {str(e)}")
            all_results[category_name] = {}
    
    # Print summary
    print(f"\n{'='*60}")
    print("CONVERSION TEST SUMMARY")
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
    
    print(f"\nTotal: {passed_tests}/{total_tests} conversions passed")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%" if total_tests > 0 else "0%")
    
    if passed_tests == total_tests:
        print("\n🎉 All conversions passed! Artifacts saved in 'conversion_artifacts' directory.")
        print("Now run 'python test_upgrade.py' in the new environment.")
        return 0
    else:
        print(f"\n❌ {total_tests - passed_tests} conversions failed")
        return 1

if __name__ == "__main__":
    # Ensure we're in the correct directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir.parent)
    
    # Check if main.py exists
    if not Path('main.py').exists():
        print("Error: main.py not found in current directory. Please run this script from the Scikit-learn-Model-Updater repository root.")
        sys.exit(1)
    
    exit_code = main()
    sys.exit(exit_code)