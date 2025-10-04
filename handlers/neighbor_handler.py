import sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
import pickle
import numpy as np

def convert_model(model, args):
    """
    Convert KNN model by extracting only the essential data needed to reconstruct it.
    We don't create a shell model at all - just store parameters and training data.
    """
    
    # Store model parameters
    model_params = {
        'class_name': model.__class__.__name__,
        'n_neighbors': getattr(model, 'n_neighbors', 5),
        'algorithm': getattr(model, 'algorithm', 'auto'),
        'leaf_size': getattr(model, 'leaf_size', 30),
        'p': getattr(model, 'p', 2),
        'metric': getattr(model, 'metric', 'minkowski'),
        'metric_params': getattr(model, 'metric_params', None),
        'n_jobs': getattr(model, 'n_jobs', None)
    }
    
    # Only add 'weights' for classifier/regressor models
    if not isinstance(model, NearestNeighbors):
        model_params['weights'] = getattr(model, 'weights', 'uniform')
    
    # Store fitted attributes
    fitted_attrs = {
        'n_features_in_': getattr(model, 'n_features_in_', None),
        'n_samples_fit_': getattr(model, 'n_samples_fit_', None),
        'effective_metric_params_': getattr(model, 'effective_metric_params_', {}),
        'n_outputs_': getattr(model, 'n_outputs_', 1),
        'classes_': getattr(model, 'classes_', None),
    }
    
    # Store the training data - this is what we'll use to refit
    training_data = {
        'X_': getattr(model, '_fit_X', None),
        'y_': getattr(model, '_y', None),
        '_fit_method': getattr(model, '_fit_method', 'auto'),
        'outputs_2d_': getattr(model, 'outputs_2d_', None)
    }
    
    # Combine everything into upgraded_state
    upgraded_state = {
        'model_params': model_params,
        'fitted_attrs': fitted_attrs,
        'training_data': training_data,
        'knn_no_shell': True  # Flag to indicate no shell needed
    }
    
    # Return upgraded_state, model_attrs (for metadata), and None for shell
    # The shell is not needed for KNN models - returning None prevents pickle issues
    return upgraded_state, model_params, None


def upgrade_model(shell, upgraded_state, model_attrs):
    """
    Upgrade KNN model by creating a fresh instance and refitting with stored data.
    The 'shell' parameter will be None for KNN models.
    """
    
    # Extract stored data
    model_params = upgraded_state.get('model_params', {})
    fitted_attrs = upgraded_state.get('fitted_attrs', {})
    training_data = upgraded_state.get('training_data', {})
    
    # Get the model class
    class_name = model_params.get('class_name')
    if class_name == 'KNeighborsClassifier':
        model_class = KNeighborsClassifier
    elif class_name == 'KNeighborsRegressor':
        model_class = KNeighborsRegressor
    elif class_name == 'NearestNeighbors':
        model_class = NearestNeighbors
    else:
        raise ValueError(f"Unknown KNN model class: {class_name}")
    
    # Create fresh model instance with stored parameters
    init_params = {
        'n_neighbors': model_params.get('n_neighbors', 5),
        'algorithm': model_params.get('algorithm', 'auto'),
        'leaf_size': model_params.get('leaf_size', 30),
        'p': model_params.get('p', 2),
        'metric': model_params.get('metric', 'minkowski'),
        'metric_params': model_params.get('metric_params'),
        'n_jobs': model_params.get('n_jobs')
    }
    
    # Add weights parameter only for classifier/regressor
    if class_name != 'NearestNeighbors':
        init_params['weights'] = model_params.get('weights', 'uniform')
    
    new_model = model_class(**init_params)
    
    # Refit the model with stored training data
    X_ = training_data.get('X_')
    y_ = training_data.get('y_')
    
    if X_ is not None:
        if y_ is not None:
            # Classifier or Regressor
            new_model.fit(X_, y_)
        else:
            # NearestNeighbors
            new_model.fit(X_)
    else:
        raise ValueError("No training data found - cannot reconstruct KNN model")
    
    # Restore any additional fitted attributes that might not be set by fit()
    for attr_name, attr_value in fitted_attrs.items():
        if attr_value is not None and not hasattr(new_model, attr_name):
            setattr(new_model, attr_name, attr_value)
    
    # Set additional attributes
    outputs_2d_ = training_data.get('outputs_2d_')
    if outputs_2d_ is not None:
        new_model.outputs_2d_ = outputs_2d_
    
    fit_method = training_data.get('_fit_method')
    if fit_method is not None:
        new_model._fit_method = fit_method
    
    return new_model