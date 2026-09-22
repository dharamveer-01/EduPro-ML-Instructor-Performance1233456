
import os
import json
import pickle
import numpy as np
import pandas as pd


class EduProModelServer:

    def __init__(
        self,
        base_dir="/content/edupro_ml_system"
    ):

        self.base_dir = base_dir

        self.model_dir = os.path.join(
            base_dir,
            "models"
        )

        self.serving_dir = os.path.join(
            base_dir,
            "serving"
        )

        self.models = {}

        self.schemas = {}

        self.load_schemas()

        self.load_models()


    # ========================================================
    # LOAD SCHEMAS
    # ========================================================

    def load_schemas(self):

        paths = {

            "instructor":
                os.path.join(
                    self.serving_dir,
                    "instructor_feature_schema.json"
                ),

            "course":
                os.path.join(
                    self.serving_dir,
                    "course_feature_schema.json"
                )
        }

        for name, path in paths.items():

            if os.path.exists(path):

                with open(
                    path,
                    "r"
                ) as f:

                    self.schemas[
                        name
                    ] = json.load(f)


    # ========================================================
    # LOAD MODELS
    # ========================================================

    def load_models(self):

        paths = {

            "instructor":
                os.path.join(
                    self.model_dir,
                    "ensemble",
                    "best_instructor_ensemble_model.pkl"
                ),

            "course":
                os.path.join(
                    self.model_dir,
                    "ensemble",
                    "best_course_ensemble_model.pkl"
                )
        }

        for name, path in paths.items():

            if os.path.exists(path):

                try:

                    with open(
                        path,
                        "rb"
                    ) as f:

                        self.models[
                            name
                        ] = pickle.load(f)

                except Exception:

                    self.models[
                        name
                    ] = None


    # ========================================================
    # ALIGN INPUT
    # ========================================================

    def align(
        self,
        entity,
        data
    ):

        if entity not in self.schemas:

            raise ValueError(
                f"Unknown entity: {entity}"
            )

        schema = self.schemas[
            entity
        ]

        expected = schema[
            "features"
        ]

        medians = schema[
            "medians"
        ]

        if isinstance(
            data,
            dict
        ):

            frame = pd.DataFrame(
                [data]
            )

        else:

            frame = pd.DataFrame(
                data
            )

        output = pd.DataFrame(
            index=frame.index
        )

        for feature in expected:

            if feature in frame.columns:

                output[
                    feature
                ] = pd.to_numeric(
                    frame[feature],
                    errors="coerce"
                )

            else:

                output[
                    feature
                ] = medians.get(
                    feature,
                    0.0
                )

        for feature in expected:

            output[
                feature
            ] = (
                output[
                    feature
                ]
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
                .fillna(
                    medians.get(
                        feature,
                        0.0
                    )
                )
            )

        return output[
            expected
        ]


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        entity,
        data
    ):

        if entity not in self.models:

            raise ValueError(
                f"Model unavailable: {entity}"
            )

        model = self.models[
            entity
        ]

        if model is None:

            raise ValueError(
                f"Model unavailable: {entity}"
            )

        X = self.align(
            entity,
            data
        )

        prediction = model.predict(
            X
        )

        return prediction.tolist()


    # ========================================================
    # MODEL STATUS
    # ========================================================

    def status(self):

        return {

            entity:
                model is not None

            for entity, model
            in self.models.items()
        }
