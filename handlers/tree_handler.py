import numpy as np
from sklearn.tree._tree import Tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def upgrade_tree_state(tree_state):
    """
    Upgrade tree state by handling the different structures in various scikit-learn versions.
    The tree state is typically a dictionary-like object with numpy arrays.
    """
    # Check if tree_state is a dictionary-like object (which it should be)
    if hasattr(tree_state, '__getitem__') and hasattr(tree_state, 'keys'):
        # This is likely a dictionary-like state object
        if 'nodes' in tree_state:
            nodes = tree_state['nodes']
            shape = nodes.shape
            dtype = nodes.dtype
            
            # Check if 'missing_go_to_left' field already exists
            if 'missing_go_to_left' in dtype.names:
                return tree_state
            
            # Add the missing_go_to_left field for newer sklearn versions
            new_dtype = np.dtype(dtype.descr + [("missing_go_to_left", "|u1")])
            new_nodes = np.empty(shape, dtype=new_dtype)
            
            # Copy existing fields
            for field_name in dtype.names:
                new_nodes[field_name] = nodes[field_name]
            
            # Initialize the new field
            new_nodes["missing_go_to_left"] = np.zeros(shape[0], dtype=np.uint8)
            
            # Update the nodes in the tree_state
            tree_state['nodes'] = new_nodes
            return tree_state
        else:
            # If it's a tuple-like structure (old format), handle that
            if isinstance(tree_state, (list, tuple)) and len(tree_state) > 0:
                nodes = tree_state[0]
                if hasattr(nodes, 'shape') and hasattr(nodes, 'dtype'):
                    shape = nodes.shape
                    dtype = nodes.dtype
                    
                    # Check if 'missing_go_to_left' field already exists
                    if 'missing_go_to_left' in dtype.names:
                        return tree_state
                    
                    # Add the missing_go_to_left field for newer sklearn versions
                    new_dtype = np.dtype(dtype.descr + [("missing_go_to_left", "|u1")])
                    new_nodes = np.empty(shape, dtype=new_dtype)
                    
                    # Copy existing fields
                    for field_name in dtype.names:
                        new_nodes[field_name] = nodes[field_name]
                    
                    # Initialize the new field
                    new_nodes["missing_go_to_left"] = np.zeros(shape[0], dtype=np.uint8)
                    
                    # Return the updated tree state with the new nodes
                    if isinstance(tree_state, tuple):
                        updated_tree_state = (new_nodes,) + tree_state[1:]
                    else:  # list
                        updated_tree_state = [new_nodes] + tree_state[1:]
                    
                    return updated_tree_state
    elif isinstance(tree_state, (list, tuple)):
        # Handle tuple format (old sklearn versions)
        if len(tree_state) > 0:
            nodes = tree_state[0]
            if hasattr(nodes, 'shape') and hasattr(nodes, 'dtype'):
                shape = nodes.shape
                dtype = nodes.dtype
                
                # Check if 'missing_go_to_left' field already exists
                if 'missing_go_to_left' in dtype.names:
                    return tree_state
                
                # Add the missing_go_to_left field for newer sklearn versions
                new_dtype = np.dtype(dtype.descr + [("missing_go_to_left", "|u1")])
                new_nodes = np.empty(shape, dtype=new_dtype)
                
                # Copy existing fields
                for field_name in dtype.names:
                    new_nodes[field_name] = nodes[field_name]
                
                # Initialize the new field
                new_nodes["missing_go_to_left"] = np.zeros(shape[0], dtype=np.uint8)
                
                # Return the updated tree state with the new nodes
                if isinstance(tree_state, tuple):
                    updated_tree_state = (new_nodes,) + tree_state[1:]
                else:  # list
                    updated_tree_state = [new_nodes] + tree_state[1:]
                
                return updated_tree_state
    
    # If we can't handle the format, return as is
    return tree_state

def reconstruct_tree(tree_state, n_features, n_classes, n_outputs):
    # Handle the case where n_classes might be 0 (for regressors) or an empty array
    if isinstance(n_classes, (int, np.integer)) and n_classes == 0:
        # For regressors, n_classes should be 1
        n_classes = 1
    elif isinstance(n_classes, (list, np.ndarray)):
        if len(n_classes) == 0:
            # For regressors, n_classes should be 1
            n_classes = np.array([1], dtype=np.int64)
        else:
            # Ensure it's the right format
            n_classes = np.array(n_classes, dtype=np.int64)
    else:
        # Make sure n_classes is an array with proper dtype
        n_classes = np.array([n_classes] if isinstance(n_classes, (int, np.integer)) else n_classes, dtype=np.int64)
    
    tree = Tree(n_features, n_classes, n_outputs)
    tree.__setstate__(tree_state)
    return tree

def convert_model(model, args):
    if isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
        tree = model.tree_
        tree_state = tree.__getstate__()

        upgraded_tree_state = upgrade_tree_state(tree_state)

        return upgraded_tree_state, {
            "n_features": tree.n_features,
            "n_classes": tree.n_classes,
            "n_outputs": tree.n_outputs
        }, "tree_"

    elif isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
        all_tree_states = []
        for estimator in model.estimators_:
            tree_state = estimator.tree_.__getstate__()
            upgraded_tree_state = upgrade_tree_state(tree_state)
            all_tree_states.append(upgraded_tree_state)

        return all_tree_states, {
            "n_features": model.n_features_in_,
            "n_classes": model.n_classes_ if hasattr(model, 'n_classes_') else 1,  # For regressors
            "n_outputs": model.n_outputs_
        }, "estimators_"
    else:
        return None

def upgrade_model(shell, upgraded_state, model_attrs):
    n_features = model_attrs["n_features"]
    n_classes = model_attrs["n_classes"]
    n_outputs = model_attrs["n_outputs"]

    if isinstance(shell, (DecisionTreeClassifier, DecisionTreeRegressor)):
        tree = reconstruct_tree(upgraded_state, n_features, n_classes, n_outputs)
        shell.tree_ = tree
    elif isinstance(shell, (RandomForestClassifier, RandomForestRegressor)):
        estimators = []
        for tree_state in upgraded_state:
            tree = reconstruct_tree(tree_state, n_features, n_classes, n_outputs)

            # Create a new estimator shell with the same type as the original
            if isinstance(shell, RandomForestClassifier):
                estimator_shell = DecisionTreeClassifier()
            else:  # RandomForestRegressor
                estimator_shell = DecisionTreeRegressor()

            # Get the original shell's parameters safely
            try:
                params = shell.get_params()
                # Remove parameters that don't exist in the shell
                for attr, value in params.items():
                    if hasattr(estimator_shell, attr):
                        setattr(estimator_shell, attr, value)
            except AttributeError:
                # If get_params fails, use basic parameters
                pass

            estimator_shell.n_features_in_ = n_features
            estimator_shell.n_outputs_ = n_outputs
            if isinstance(shell, RandomForestClassifier):
                estimator_shell.classes_ = np.arange(n_classes) if n_classes > 0 else np.array([0])

            estimator_shell.tree_ = tree
            estimators.append(estimator_shell)
        
        shell.estimators_ = estimators
    return shell