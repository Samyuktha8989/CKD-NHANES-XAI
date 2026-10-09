"""
Leakage-safe preprocessing pipeline builder and feature name tracking module.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

deps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "deps"))
if os.path.exists(deps_path) and deps_path not in sys.path:
    sys.path.insert(0, deps_path)


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    scale_numeric: bool = True
) -> ColumnTransformer:
    """
    Build a leakage-safe ColumnTransformer.
    Numeric: SimpleImputer(median) [+ StandardScaler]
    Categorical: SimpleImputer(most_frequent) + OneHotEncoder(ignore unknown)
    """
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=num_steps)

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", cat_pipeline, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
    return preprocessor


def get_feature_names_and_mapping(
    fitted_preprocessor: ColumnTransformer,
    numeric_features: List[str],
    categorical_features: List[str]
) -> Tuple[List[str], Dict[str, str]]:
    """
    Extract transformed feature names from a fitted preprocessor and construct
    a mapping from each transformed column name back to its original parent feature.
    """
    transformed_feature_names = []
    column_to_original_map = {}

    # Numeric names
    for num_col in numeric_features:
        name = f"num__{num_col}"
        transformed_feature_names.append(name)
        column_to_original_map[name] = num_col

    # Categorical names from OneHotEncoder
    if categorical_features and "cat" in fitted_preprocessor.named_transformers_:
        cat_transformer = fitted_preprocessor.named_transformers_["cat"]
        onehot_step = cat_transformer.named_steps["onehot"]
        cat_names = onehot_step.get_feature_names_out(categorical_features)
        for cat_col_encoded in cat_names:
            name = f"cat__{cat_col_encoded}"
            transformed_feature_names.append(name)
            # Find the original feature prefix
            for orig in categorical_features:
                if cat_col_encoded.startswith(orig):
                    column_to_original_map[name] = orig
                    break

    return transformed_feature_names, column_to_original_map
