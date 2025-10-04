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

        # Handle different pipeline types with their specific attributes
        if isinstance(model, Pipeline):
            steps_attr = 'steps'
            components = model.steps
        elif isinstance(model, FeatureUnion):
            steps_attr = 'transformer_list'
            components = model.transformer_list
        elif isinstance(model, ColumnTransformer):
            steps_attr = 'transformers'
            components = model.transformers
        else:
            return None

        for item in components:
            if isinstance(model, ColumnTransformer):
                # ColumnTransformer has format: (name, transformer, columns)
                name, component = item[0], item[1]
            else:
                # Pipeline and FeatureUnion have format: (name, transformer)
                name, component = item
            
            handler = get_handler(component)
            if not handler:
                raise TypeError(f"No handler for component {name} of type {type(component)} in pipeline.")

            # Recursively convert the component
            upgraded_state, model_attrs, attr_to_del = handler.convert_model(component, args)

            # We need to save the original component's shell as well
            if attr_to_del and hasattr(component, attr_to_del):
                delattr(component, attr_to_del)

            step_data = {
                "name": name,
                "shell": component,
                "upgraded_state": upgraded_state,
                "model_attrs": model_attrs
            }
            
            # For ColumnTransformer, we also need to preserve the columns specification
            if isinstance(model, ColumnTransformer):
                step_data["columns"] = item[2]  # The third element is the column specification
            
            steps_data.append(step_data)

        # The pipeline's own state is the collection of its converted steps
        upgraded_state = steps_data
        model_attrs = {
            'class_name': model.__class__.__name__,
            'steps_attr': steps_attr  # Store which attribute was used
        }

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
        
        if class_name == 'ColumnTransformer':
            # For ColumnTransformer, include the columns specification
            columns = step_data.get('columns')
            reconstructed_steps.append((step_data['name'], reconstructed_component, columns))
        else:
            # For Pipeline and FeatureUnion
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