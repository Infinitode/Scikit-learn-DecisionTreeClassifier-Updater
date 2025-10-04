from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors

def convert_model(model, args):
    """
    Converts a nearest neighbor model by extracting its attributes.
    """
    if isinstance(model, (KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors)):
        model_attrs = {attr: getattr(model, attr) for attr in model.__dict__ if attr.endswith('_')}

        # Add the class name for reconstruction
        model_attrs['class_name'] = model.__class__.__name__

        # The "upgraded_state" is the dictionary of attributes.
        upgraded_state = model_attrs

        return upgraded_state, model_attrs, None  # No attribute to delete
    return None

def upgrade_model(shell, upgraded_state, model_attrs):
    """
    Upgrades a nearest neighbor model by creating a new instance and setting its attributes.
    """
    class_name = model_attrs.get('class_name')
    if not class_name:
        raise ValueError("Neighbor model class name not found in model attributes.")

    # Create a new instance
    if class_name == 'KNeighborsClassifier':
        new_model = KNeighborsClassifier()
    elif class_name == 'KNeighborsRegressor':
        new_model = KNeighborsRegressor()
    elif class_name == 'NearestNeighbors':
        new_model = NearestNeighbors()
    else:
        raise TypeError(f"Unsupported neighbor model type: {class_name}")

    # Set the attributes on the new model
    for attr, value in upgraded_state.items():
        setattr(new_model, attr, value)

    return new_model