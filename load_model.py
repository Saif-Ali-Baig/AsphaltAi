import pickle
import xgboost as xgb

def load_model(path, model_type='sklearn'):
    if model_type == 'xgboost':
        model = xgb.XGBClassifier()
        model.load_model(path)
        return model
    elif model_type == 'sklearn':
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError("Unsupported model type.")
