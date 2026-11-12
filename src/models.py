"""
src/models.py
=============
Model factory for the Ahmad et al. (2020) replication.

Creates every classifier described in the paper with:
  - Reproducible random_state
  - Grid search over small, sensible parameter grids
  - 10-fold cross-validation for ensemble learners

All functions return scikit-learn–compatible estimators.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import (
    PARAM_GRIDS,
    RANDOM_STATE,
    CV_FOLDS,
    GRID_SEARCH_N_JOBS,
    GRID_SEARCH_PRE_DISPATCH,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Individual learners
# ---------------------------------------------------------------------------

def make_lr(params: Optional[Dict] = None) -> LogisticRegression:
    """Logistic Regression (LR)."""
    p = params or {"C": 1.0, "solver": "lbfgs", "max_iter": 1000}
    return LogisticRegression(random_state=RANDOM_STATE, **p)


def make_lsvm(params: Optional[Dict] = None) -> LinearSVC:
    """Linear Support Vector Machine (LSVM)."""
    p = params or {"C": 1.0, "max_iter": 2000}
    return LinearSVC(random_state=RANDOM_STATE, dual="auto", **p)


def make_mlp(params: Optional[Dict] = None) -> MLPClassifier:
    """Multilayer Perceptron (MLP)."""
    p = params or {"hidden_layer_sizes": (100,), "max_iter": 200, "early_stopping": True}
    return MLPClassifier(random_state=RANDOM_STATE, **p)


def make_knn(params: Optional[Dict] = None) -> KNeighborsClassifier:
    """K-Nearest Neighbours (KNN) — n_jobs=1 prevents nested threads under GridSearchCV."""
    p = params or {"n_neighbors": 5, "metric": "euclidean"}
    return KNeighborsClassifier(n_jobs=1, **p)


# ---------------------------------------------------------------------------
# Ensemble learners
# ---------------------------------------------------------------------------

def make_rf(params: Optional[Dict] = None) -> RandomForestClassifier:
    """Random Forest (100 estimators, Gini criterion) — paper §2.3.1. n_jobs=1 avoids thread explosion."""
    p = params or {"n_estimators": 100, "criterion": "gini", "max_features": "sqrt"}
    return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **p)


def make_bagging(params: Optional[Dict] = None) -> BaggingClassifier:
    """Bagging with 100 Decision Trees — paper §2.3.2. Runs standalone with controlled n_jobs=2."""
    p = params or {"n_estimators": 100, "max_samples": 0.8}
    base = DecisionTreeClassifier(random_state=RANDOM_STATE)
    return BaggingClassifier(
        estimator=base,
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=2,
        **p,
    )


def make_adaboost(params: Optional[Dict] = None) -> AdaBoostClassifier:
    """AdaBoost over Decision Trees — paper §2.3.3."""
    p = params or {"n_estimators": 50, "learning_rate": 1.0}
    base = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE)
    return AdaBoostClassifier(estimator=base, random_state=RANDOM_STATE, **p)


def make_xgb(params: Optional[Dict] = None) -> XGBClassifier:
    """XGBoost Classifier — paper §2.3.4. n_jobs=1 prevents OpenMP thread proliferation."""
    p = params or {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6}
    return XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,
        eval_metric="logloss",
        verbosity=0,
        **p,
    )


def make_voting1(lr, rf, knn) -> VotingClassifier:
    """
    Voting Classifier 1: Majority vote of LR, RF, KNN.
    Assumes the three base learners are already fitted.
    """
    vc = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("knn", knn)],
        voting="hard",
    )
    return vc


def make_voting2(lr, lsvm, dt) -> VotingClassifier:
    """
    Voting Classifier 2: Majority vote of LR, LSVM, Decision Tree.
    Assumes the three base learners are already fitted.
    """
    vc = VotingClassifier(
        estimators=[("lr", lr), ("lsvm", lsvm), ("dt", dt)],
        voting="hard",
    )
    return vc


# ---------------------------------------------------------------------------
# Grid search wrapper
# ---------------------------------------------------------------------------

def grid_search(
    estimator,
    param_grid: Dict[str, Any],
    X_train,
    y_train,
    cv: int = CV_FOLDS,
    scoring: str = "accuracy",
    n_jobs: int = GRID_SEARCH_N_JOBS,
    pre_dispatch: str = GRID_SEARCH_PRE_DISPATCH,
) -> Any:
    """
    Run GridSearchCV on *estimator* using *param_grid*.

    Parameters
    ----------
    cv : int
        Number of cross-validation folds (paper uses 10).
    scoring : str
        sklearn scoring metric.
    n_jobs : int
        Controlled worker count (default 2 from config) to prevent RAM explosion.
    pre_dispatch : str
        Limits queue task buffering to conserve memory.

    Returns
    -------
    Best fitted estimator, best parameters, and best CV score.
    """
    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        pre_dispatch=pre_dispatch,
        refit=True,
    )
    gs.fit(X_train, y_train)
    logger.info(
        f"  GridSearch best params: {gs.best_params_} | "
        f"CV {scoring}: {gs.best_score_:.4f}"
    )
    return gs.best_estimator_, gs.best_params_, gs.best_score_


# ---------------------------------------------------------------------------
# Catalogue of all models (used by train_models.py)
# ---------------------------------------------------------------------------

MODEL_CATALOGUE = {
    # (category,   display_name,                     factory_fn,    param_grid_key)
    "LR":      ("Individual", "Logistic Regression (LR)",        make_lr,        "LR"),
    "LSVM":    ("Individual", "Linear SVM (LSVM)",               make_lsvm,      "LSVM"),
    "MLP":     ("Individual", "Multilayer Perceptron (MLP)",      make_mlp,       "MLP"),
    "KNN":     ("Individual", "K-Nearest Neighbors (KNN)",       make_knn,       "KNN"),
    "RF":      ("Ensemble",   "Random Forest (RF)",               make_rf,        "RF"),
    "Bagging": ("Ensemble",   "Bagging Classifier (DTs)",         make_bagging,   "Bagging"),
    "AdaBoost":("Ensemble",   "Boosting Classifier (AdaBoost)",   make_adaboost,  "AdaBoost"),
    "XGB":     ("Ensemble",   "Boosting Classifier (XGBoost)",    make_xgb,       "XGB"),
    # Voting classifiers are built after their base learners are trained
}

# Ordered list for results tables
ORDERED_MODEL_KEYS = [
    "LR", "LSVM", "MLP", "KNN",
    "RF", "VC1", "VC2", "Bagging", "AdaBoost", "XGB",
]
