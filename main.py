import argparse
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
import utils
from handlers import (
    tree_handler, scaler_handler, neighbor_handler, svm_handler,
    linear_model_handler, decomposition_handler, preprocessing_handler,
    pipeline_handler
)

MODEL_HANDLERS = {
    (DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier, RandomForestRegressor): tree_handler,
    (StandardScaler, MinMaxScaler): scaler_handler,
    (KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors): neighbor_handler,
    (SVC, SVR): svm_handler,
    (LogisticRegression, Ridge, Lasso): linear_model_handler,
    (PCA, IncrementalPCA, KernelPCA): decomposition_handler,
    (PolynomialFeatures, OneHotEncoder): preprocessing_handler,
    (Pipeline, FeatureUnion, ColumnTransformer): pipeline_handler
}

# Provide the pipeline handler with access to all other handlers
pipeline_handler.set_handlers(MODEL_HANDLERS)

def get_handler(model):
    for model_types, handler in MODEL_HANDLERS.items():
        if isinstance(model, model_types):
            return handler
    return None

def convert_model(args):
    model = utils.load_model(args.model_path)
    handler = get_handler(model)

    if not handler:
        raise TypeError(f"Model type {type(model)} not supported.")

    result = handler.convert_model(model, args)
    if result:
        upgraded_state, model_attrs, attr_to_del = result
        utils.save_model(upgraded_state, args.upgraded_tree_state_path)
        utils.save_json(model_attrs, args.model_json_path)

        if attr_to_del and hasattr(model, attr_to_del):
            delattr(model, attr_to_del)
        utils.save_model(model, args.shell_path)
    else:
        raise TypeError(f"Conversion failed for model type {type(model)}.")


def upgrade_model(args):
    shell = utils.load_model(args.shell_path)
    handler = get_handler(shell)

    if not handler:
        # For models like scalers, the shell might not be enough to determine the handler.
        # We can get the class name from the json file.
        model_attrs = utils.load_json(args.model_json_path)
        class_name = model_attrs.get('class_name')
        if class_name:
            # This is a bit of a hack, but it works for now.
            # We create a dummy instance of the class to get the handler.
            if class_name == 'StandardScaler':
                handler = get_handler(StandardScaler())
            elif class_name == 'MinMaxScaler':
                handler = get_handler(MinMaxScaler())
            elif class_name == 'KNeighborsClassifier':
                handler = get_handler(KNeighborsClassifier())
            elif class_name == 'KNeighborsRegressor':
                handler = get_handler(KNeighborsRegressor())
            elif class_name == 'NearestNeighbors':
                handler = get_handler(NearestNeighbors())
            elif class_name == 'SVC':
                handler = get_handler(SVC())
            elif class_name == 'SVR':
                handler = get_handler(SVR())
            elif class_name == 'LogisticRegression':
                handler = get_handler(LogisticRegression())
            elif class_name == 'Ridge':
                handler = get_handler(Ridge())
            elif class_name == 'Lasso':
                handler = get_handler(Lasso())
            elif class_name == 'PCA':
                handler = get_handler(PCA())
            elif class_name == 'IncrementalPCA':
                handler = get_handler(IncrementalPCA())
            elif class_name == 'KernelPCA':
                handler = get_handler(KernelPCA())
            elif class_name == 'PolynomialFeatures':
                handler = get_handler(PolynomialFeatures())
            elif class_name == 'OneHotEncoder':
                handler = get_handler(OneHotEncoder())
            elif class_name == 'Pipeline':
                handler = get_handler(Pipeline(steps=[]))
            elif class_name == 'FeatureUnion':
                handler = get_handler(FeatureUnion(transformer_list=[]))
            elif class_name == 'ColumnTransformer':
                handler = get_handler(ColumnTransformer(transformers=[]))

    if not handler:
        raise TypeError(f"Model type {type(shell)} not supported for upgrade.")

    upgraded_state = utils.load_model(args.upgraded_tree_state_path)
    model_attrs = utils.load_json(args.model_json_path)

    upgraded_model = handler.upgrade_model(shell, upgraded_state, model_attrs)
    utils.save_model(upgraded_model, args.output_path)


def main():
    parser = argparse.ArgumentParser(description="Upgrade Scikit-learn models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Converter parser
    parser_convert = subparsers.add_parser("convert", help="Convert a model to be upgradeable.")
    parser_convert.add_argument("--model_path", required=True, help="Path to the original model file.")
    parser_convert.add_argument("--shell_path", required=True, help="Path to save the model shell.")
    parser_convert.add_argument("--upgraded_tree_state_path", required=True, help="Path to save the upgraded tree state.")
    parser_convert.add_argument("--model_json_path", required=True, help="Path to save the model metadata as JSON.")
    parser_convert.set_defaults(func=convert_model)

    # Upgrader parser
    parser_upgrade = subparsers.add_parser("upgrade", help="Upgrade a converted model.")
    parser_upgrade.add_argument("--shell_path", required=True, help="Path to the model shell file.")
    parser_upgrade.add_argument("--upgraded_tree_state_path", required=True, help="Path to the upgraded tree state file.")
    parser_upgrade.add_argument("--model_json_path", required=True, help="Path to the model metadata JSON file.")
    parser_upgrade.add_argument("--output_path", required=True, help="Path to save the final upgraded model.")
    parser_upgrade.set_defaults(func=upgrade_model)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()