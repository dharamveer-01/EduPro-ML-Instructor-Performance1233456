
import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path


class ExactModelServer:

    def __init__(self, base_dir):

        self.base_dir = Path(base_dir)

        self.registry_path = (
            self.base_dir
            / "model_registry"
            / "exact_model_serving_registry.json"
        )

        self.registry = None
        self.models = {}
        self.schemas = {}

        self.load_registry()
        self.load_models()
        self.load_schemas()


    def load_registry(self):

        if not self.registry_path.exists():

            raise FileNotFoundError(
                f"Registry not found: {self.registry_path}"
            )

        with open(self.registry_path, "r") as f:

            self.registry = json.load(f)


    def load_models(self):

        for entity in ["instructor", "course"]:

            info = self.registry[
                "models"
            ].get(entity, {})

            model_path = info.get(
                "selected_model"
            )

            if not model_path:

                self.models[entity] = None
                continue

            path = Path(model_path)

            if not path.exists():

                raise FileNotFoundError(
                    f"{entity} model not found: {path}"
                )

            with open(path, "rb") as f:

                self.models[entity] = pickle.load(f)


    def load_schemas(self):

        for entity in ["instructor", "course"]:

            info = self.registry[
                "models"
            ].get(entity, {})

            schema_path = info.get(
                "feature_schema"
            )

            if not schema_path:

                raise FileNotFoundError(
                    f"No schema registered for {entity}"
                )

            path = Path(schema_path)

            if not path.exists():

                raise FileNotFoundError(
                    f"Schema not found: {path}"
                )

            with open(path, "r") as f:

                self.schemas[entity] = json.load(f)


    def status(self):

        result = {}

        for entity in ["instructor", "course"]:

            model = self.models.get(entity)
            schema = self.schemas.get(entity)

            result[entity] = {

                "model_loaded":
                    model is not None,

                "model_class":
                    type(model).__name__
                    if model is not None
                    else None,

                "feature_count":
                    schema.get("feature_count")
                    if schema
                    else 0,

                "feature_order":
                    schema.get("feature_order", [])
                    if schema
                    else [],

                "schema_version":
                    schema.get("schema_version")
                    if schema
                    else None,

                "model_path":
                    self.registry[
                        "models"
                    ][entity].get(
                        "selected_model"
                    )
            }

        return result


    def align_features(
        self,
        entity,
        features
    ):

        if entity not in self.schemas:

            raise ValueError(
                f"Unknown entity: {entity}"
            )

        schema = self.schemas[entity]

        required = schema.get(
            "feature_order",
            []
        )

        defaults = schema.get(
            "feature_defaults",
            {}
        )

        if not required:

            raise ValueError(
                f"No features registered for {entity}"
            )

        if not isinstance(features, dict):

            raise ValueError(
                "Features must be a dictionary"
            )

        row = {}

        for feature in required:

            value = features.get(
                feature,
                defaults.get(feature, 0.0)
            )

            try:

                value = float(value)

            except Exception:

                value = float(
                    defaults.get(
                        feature,
                        0.0
                    )
                )

            if not np.isfinite(value):

                value = float(
                    defaults.get(
                        feature,
                        0.0
                    )
                )

            row[feature] = value

        X = pd.DataFrame(
            [row],
            columns=required
        )

        return X


    def predict(
        self,
        entity,
        features
    ):

        model = self.models.get(entity)

        if model is None:

            raise RuntimeError(
                f"No model loaded for {entity}"
            )

        X = self.align_features(
            entity,
            features
        )

        prediction = model.predict(X)

        value = float(
            np.asarray(
                prediction
            ).reshape(-1)[0]
        )

        return {
            "entity": entity,
            "prediction": value,
            "feature_count": X.shape[1],
            "feature_order": X.columns.tolist()
        }


    def batch_predict(
        self,
        entity,
        records
    ):

        if not isinstance(records, list):

            raise ValueError(
                "records must be a list"
            )

        if len(records) > 500:

            raise ValueError(
                "Maximum batch size is 500"
            )

        model = self.models.get(entity)

        if model is None:

            raise RuntimeError(
                f"No model loaded for {entity}"
            )

        X_rows = []

        for record in records:

            X = self.align_features(
                entity,
                record
            )

            X_rows.append(
                X.iloc[0].to_dict()
            )

        X_batch = pd.DataFrame(
            X_rows,
            columns=self.schemas[
                entity
            ]["feature_order"]
        )

        predictions = model.predict(
            X_batch
        )

        output = []

        for i, value in enumerate(
            np.asarray(predictions).reshape(-1)
        ):

            output.append({
                "row": i,
                "prediction": float(value)
            })

        return {
            "entity": entity,
            "count": len(output),
            "predictions": output
        }
