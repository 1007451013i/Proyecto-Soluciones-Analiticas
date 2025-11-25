from typing import Optional, Tuple

import pandas as pd

from model.config.core import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    validated_data = input_data.copy()
    # If features are defined, ensure we drop NA only on them; else drop none
    features = config.model_config.features or list(validated_data.columns)
    validated_data.dropna(subset=features, inplace=True)
    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    relevant_data = input_data.copy()
    # If features are defined, subset
    if config.model_config.features:
        # only keep expected features
        relevant_data = relevant_data[config.model_config.features].copy()

    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    # We keep validation permissive to allow MLflow model signature to enforce types
    return validated_data, errors
