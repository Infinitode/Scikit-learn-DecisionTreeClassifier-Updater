from sklearn.preprocessing import StandardScaler, MinMaxScaler

def convert_model(model, args):
    """
    Converts a scaler model by extracting its attributes.
    For scalers, the 'upgraded_state' is simply a dictionary of their attributes.
    The 'shell' is not needed as we can reconstruct the scaler from scratch.
    """
    if isinstance(model, (StandardScaler, MinMaxScaler)):
        model_attrs = {attr: getattr(model, attr) for attr in model.__dict__ if attr.endswith('_')}

        # The "upgraded_state" for scalers is just their attributes.
        # We don't have a complex state to upgrade like with trees.
        upgraded_state = model_attrs

        # We will return the name of the class to be able to reconstruct it later
        model_attrs['class_name'] = model.__class__.__name__

        return upgraded_state, model_attrs, None # No attribute to delete
    return None

def upgrade_model(shell, upgraded_state, model_attrs):
    """
    Upgrades a scaler model by creating a new instance and setting its attributes.
    The 'shell' is the original model object, but it's not strictly necessary
    as we can determine the class from model_attrs.
    """
    class_name = model_attrs.get('class_name')
    if not class_name:
        raise ValueError("Scaler class name not found in model attributes.")

    # Create a new scaler instance
    if class_name == 'StandardScaler':
        new_scaler = StandardScaler()
    elif class_name == 'MinMaxScaler':
        new_scaler = MinMaxScaler()
    else:
        raise TypeError(f"Unsupported scaler type: {class_name}")

    # Set the attributes on the new scaler
    for attr, value in upgraded_state.items():
        setattr(new_scaler, attr, value)

    return new_scaler