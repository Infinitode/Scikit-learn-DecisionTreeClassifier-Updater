import argparse
import pickle
import numpy
from sklearn.tree._tree import Tree

def main():
    parser = argparse.ArgumentParser(description="Upgrade Scikit-learn DecisionTreeClassifier models to newer versions.")
    parser.add_argument("--shell_path", required=True, help="Path to the classifier shell file (without the tree)")
    parser.add_argument("--upgraded_tree_state_path", required=True, help="Path to the upgraded tree state file")
    parser.add_argument("--model_txt_path", required=True, help="Path to the model text file")
    parser.add_argument("--output_path", required=True, help="Path to save the final upgraded classifier")

    args = parser.parse_args()

    # Load the classifier shell
    with open(args.shell_path, "rb") as f:
        classifier = pickle.load(f)

    # Load the upgraded tree state
    with open(args.upgraded_tree_state_path, "rb") as f:
        tree_state = pickle.load(f)

    with open(args.model_txt_path, 'r') as f:
        n_features = int(f.readline().strip())
        n_classes = numpy.array(eval(f.readline().strip()))  # Convert the string back to a list/array
        n_outputs = int(f.readline().strip())

    # Ensure all required attributes are present and valid
    if not isinstance(n_features, int) or n_features <= 0:
        raise ValueError("Invalid or missing 'n_features' in the tree state.")
    if not isinstance(n_classes, (list, numpy.ndarray)) or len(n_classes) == 0:
        raise ValueError("Invalid or missing 'n_classes' in the tree state.")
    if not isinstance(n_outputs, int) or n_outputs <= 0:
        raise ValueError("Invalid or missing 'n_outputs' in the tree state.")

    # Recreate the tree with updated state
    tree = Tree(n_features=n_features, n_classes=numpy.asarray(n_classes), n_outputs=n_outputs)
    tree.__setstate__(tree_state)

    # Add the monotonic_cst attribute for compatibility if missing
    if not hasattr(classifier, "monotonic_cst"):
        classifier.monotonic_cst = None

    # Reassign the tree attribute
    classifier.tree_ = tree

    # Save the upgraded classifier
    with open(args.output_path, "wb") as f:
        pickle.dump(classifier, f)

if __name__ == "__main__":
    main()