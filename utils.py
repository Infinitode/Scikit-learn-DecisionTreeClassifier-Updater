import pickle
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        if obj is None:
            return None
        return json.JSONEncoder.default(self, obj)

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, cls=NumpyEncoder)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
