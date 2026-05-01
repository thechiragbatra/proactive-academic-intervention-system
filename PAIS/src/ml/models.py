from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .. import config as C


def build_preprocessor(numeric_cols, categorical_cols):
    num = Pipeline([
        ("imp",   SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat = Pipeline([
        ("imp",    SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num, numeric_cols),
        ("cat", cat, categorical_cols),
    ])


def build_pipelines(numeric_cols, categorical_cols):
    pre = build_preprocessor(numeric_cols, categorical_cols)
    seed = C.RANDOM_SEED
    return {
        "logistic_regression": Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=seed)),
        ]),
        "random_forest": Pipeline([
            ("pre", pre),
            ("clf", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=3,
                class_weight="balanced", n_jobs=-1, random_state=seed)),
        ]),
        "gradient_boosting": Pipeline([
            ("pre", pre),
            ("clf", GradientBoostingClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                random_state=seed)),
        ]),
    }


CANDIDATES = ["logistic_regression", "random_forest", "gradient_boosting"]
