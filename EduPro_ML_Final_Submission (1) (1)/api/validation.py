
from typing import Dict, Any


def validate_numeric_features(features: Dict[str, Any]):

    if not isinstance(features, dict):
        raise ValueError(
            "Features must be a dictionary."
        )

    cleaned = {}

    for key, value in features.items():

        if value is None:
            continue

        try:
            cleaned[str(key)] = float(value)

        except Exception:
            raise ValueError(
                f"Feature '{key}' must be numeric."
            )

    return cleaned


def validate_batch_size(
    records,
    max_batch_size=500
):

    if not isinstance(records, list):
        raise ValueError(
            "Batch input must be a list."
        )

    if len(records) == 0:
        raise ValueError(
            "Batch cannot be empty."
        )

    if len(records) > max_batch_size:
        raise ValueError(
            f"Batch size exceeds maximum allowed size "
            f"of {max_batch_size}."
        )

    return True
