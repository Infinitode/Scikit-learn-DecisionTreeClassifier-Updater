import argparse
import pickle
import numpy
import json

def main():
    parser = argparse.ArgumentParser(description="Convert and upgrade Scikit-learn DecisionTreeClassifier models.")
    parser.add_argument("--model_path", required=True, help="Path to the original model file (e.g., model.pkl)")
    parser.add_argument("--shell_path", required=True, help="Path to save the classifier shell (without the tree)")
    parser.add_argument("--tree_state_path", required=True, help="Path to save the original tree state")
    parser.add_argument("--upgraded_tree_state_path", required=True, help="Path to save the upgraded tree state")
    parser.add_argument("--model_txt_path", required=True, help="Path to save the model state")

    args = parser.parse_args()

    # Load the old model
    with open(args.model_path, "rb") as f:
        classifier = pickle.load(f)

    # Get the tree from the classifier
    tree = classifier.tree_

    # Drop the tree attribute
    delattr(classifier, "tree_")

    # Save the shell of the classifier (without the tree)
    with open(args.shell_path, "wb") as f:
        pickle.dump(classifier, f)

    with open(args.model_txt_path, "w") as file:
        file.write(f"{tree.n_features}\n")
        file.write(f"{tree.n_classes.tolist()}\n")  # Convert numpy array to list for easier processing
        file.write(f"{tree.n_outputs}\n")

    # Extract and dump Cython state
    tree_state = tree.__getstate__()
    with open(args.tree_state_path, "wb") as f:
        pickle.dump(tree_state, f)

    # Open tree state
    with open(args.tree_state_path, "rb") as f:
        tree_state = pickle.load(f)

    # Get nodes and shapes
    nodes = tree_state["nodes"]
    shape = nodes.shape
    dtype = nodes.dtype

    # Create a Scikit-Learn >= 1.3.0 compatible data type
    new_dtype = numpy.dtype(dtype.descr + [("missing_go_to_left", "|u1")])
    new_nodes = numpy.empty(shape, dtype=new_dtype)

    # Copy existing dimensions
    for field_name, field_dtype in dtype.fields.items():
        new_nodes[field_name] = nodes[field_name]

    # Append a new dimension
    new_nodes["missing_go_to_left"] = numpy.zeros(shape[0], dtype=numpy.uint8)
    tree_state["nodes"] = new_nodes

    # Save the upgraded tree state
    with open(args.upgraded_tree_state_path, "wb") as f:
        pickle.dump(tree_state, f)

if __name__ == "__main__":
    main()