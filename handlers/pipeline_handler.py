from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer

# This will be populated from main.py or a registry to avoid circular imports
MODEL_HANDLERS = {}

def set_handlers(handlers):
    global MODEL_HANDLERS
    MODEL_HANDLERS = handlers

def get_handler(model):
    for model_types, handler in MODEL_HANDLERS.items():
        if isinstance(model, model_types):
            return handler
    return None

def convert_model(model, args):
    """
    Converts a pipeline model by recursively converting its components.
    """
    if isinstance(model, (Pipeline, FeatureUnion, ColumnTransformer)):
        steps_data = []

        # Determine the attribute that holds the steps/transformers
        if isinstance(model, Pipeline):
            steps_attr = 'steps'
        else: # FeatureUnion, ColumnTransformer
            steps_attr = 'transformer_list'

        for name, component in getattr(model, steps_attr):
            handler = get_handler(component)
            if not handler:
                raise TypeError(f"No handler for component {name} of type {type(component)} in pipeline.")

            # Recursively convert the component
            upgraded_state, model_attrs, attr_to_del = handler.convert_model(component, args)

            # We need to save the original component's shell as well
            if attr_to_del and hasattr(component, attr_to_del):
                delattr(component, attr_to_del)

            steps_data.append({
                "name": name,
                "shell": component,
                "upgraded_state": upgraded_state,
                "model_attrs": model_attrs
            })

        # The pipeline's own state is the collection of its converted steps
        upgraded_state = steps_data
        model_attrs = {'class_name': model.__class__.__name__}

        return upgraded_state, model_attrs, steps_attr
    return None


def upgrade_model(shell, upgraded_state, model_attrs):
    """
    Upgrades a pipeline model by recursively upgrading its components.
    """
    class_name = model_attrs.get('class_name')
    if not class_name:
        raise ValueError("Pipeline class name not found.")

    reconstructed_steps = []
    for step_data in upgraded_state:
        component_shell = step_data['shell']
        component_upgraded_state = step_data['upgraded_state']
        component_model_attrs = step_data['model_attrs']

        handler = get_handler(component_shell)
        if not handler:
             raise TypeError(f"No handler for component shell of type {type(component_shell)}.")

        # Recursively upgrade the component
        reconstructed_component = handler.upgrade_model(component_shell, component_upgraded_state, component_model_attrs)
        reconstructed_steps.append((step_data['name'], reconstructed_component))

    # Create a new pipeline instance with the reconstructed steps
    if class_name == 'Pipeline':
        new_model = Pipeline(steps=reconstructed_steps)
    elif class_name == 'FeatureUnion':
        new_model = FeatureUnion(transformer_list=reconstructed_steps)
    elif class_name == 'ColumnTransformer':
        new_model = ColumnTransformer(transformers=reconstructed_steps)
    else:
        raise TypeError(f"Unsupported pipeline type: {class_name}")

    return new_model