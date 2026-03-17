import json

import joblib
import mlflow.pyfunc
from scipy.sparse import csr_matrix


class LightFMPyFuncModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model"])
        self.user_features_matrix = joblib.load(context.artifacts["user_features_matrix"])
        self.item_features_matrix = joblib.load(context.artifacts["item_features_matrix"])
        self.interactions_csr = csr_matrix(joblib.load(context.artifacts["interactions"]))
        with open(context.artifacts["user_id_map"]) as f:
            self.user_id_map = json.load(f)
        with open(context.artifacts["item_id_map"]) as f:
            self.item_id_map = json.load(f)

    def predict(self, context, model_input):
        raise NotImplementedError(
            "Registry model is for versioning and auditability in v1. "
            "Serving continues to use disk artifacts through the FastAPI service."
        )
