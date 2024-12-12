# Run this file in your original environment (older scikit-learn version, etc.)
import pickle
import numpy

# Path to your old model file (e.g. model.pkl)
model_path = 'path/to/model.pkl'
classifier_shell_path = 'path/to/classifier_shell.pkl'
tree_state_path = 'path/to/tree_state.pkl'
upgraded_tree_state_path = 'path/to/upgraded/tree_state.pkl'
model_txt_path = 'path/to/model.txt'

# Load the old model
with open(model_path, "rb") as f:
    classifier = pickle.load(f)

# Get the tree from the classifier
tree = classifier.tree_

# Drop the tree attribute
delattr(classifier, "tree_")

# Save the shell of the classifier (without the tree)
with open(classifier_shell_path, "wb") as f:
    pickle.dump(classifier, f)

with open(model_txt_path, "w") as file:
    file.write(f"{tree.n_features}\n")
    file.write(f"{tree.n_classes.tolist()}\n")  # Convert numpy array to list for easier processing
    file.write(f"{tree.n_outputs}\n")

# Extract and dump Cython state
tree_state = tree.__getstate__()
with open(tree_state_path, "wb") as f:
    pickle.dump(tree_state, f)

# Open tree state
with open(tree_state_path, "rb") as f:
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
with open(upgraded_tree_state_path, "wb") as f:
    pickle.dump(tree_state, f)