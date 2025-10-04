import numpy as np
from sklearn.tree._tree import Tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def upgrade_tree_state(tree_state):
    nodes, _, _ = tree_state
    shape = nodes.shape
    dtype = nodes.dtype
    new_dtype = np.dtype(dtype.descr + [("missing_go_to_left", "|u1")])
    new_nodes = np.empty(shape, dtype=new_dtype)
    for field_name, _ in dtype.fields.items():
        new_nodes[field_name] = nodes[field_name]
    new_nodes["missing_go_to_left"] = np.zeros(shape[0], dtype=np.uint8)
    tree_state = (new_nodes, tree_state[1], tree_state[2])
    return tree_state

def reconstruct_tree(tree_state, n_features, n_classes, n_outputs):
    tree = Tree(n_features, np.array(n_classes, dtype=np.int64), n_outputs)
    tree.__setstate__(tree_state)
    return tree

def convert_model(model, args):
    if isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
        tree = model.tree_
        tree_state = tree.__getstate__()

        upgraded_tree_state = upgrade_tree_state(tree_state)

        return upgraded_tree_state, {
            "n_features": tree.n_features_in_,
            "n_classes": tree.n_classes_,
            "n_outputs": tree.n_outputs_
        }, "tree_"

    elif isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
        all_tree_states = []
        for estimator in model.estimators_:
            tree_state = estimator.tree_.__getstate__()
            upgraded_tree_state = upgrade_tree_state(tree_state)
            all_tree_states.append(upgraded_tree_state)

        return all_tree_states, {
            "n_features": model.n_features_in_,
            "n_classes": model.n_classes_,
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

            estimator_shell = DecisionTreeClassifier() if isinstance(shell, RandomForestClassifier) else DecisionTreeRegressor()

            for attr, value in shell.get_params().items():
                if hasattr(estimator_shell, attr):
                    setattr(estimator_shell, attr, value)

            estimator_shell.n_features_in_ = n_features
            estimator_shell.n_outputs_ = n_outputs
            estimator_shell.classes_ = np.arange(n_classes) if isinstance(shell, RandomForestClassifier) else None

            estimator_shell.tree_ = tree
            estimators.append(estimator_shell)
        shell.estimators_ = estimators
    return shell