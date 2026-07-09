from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


@dataclass
class _ModelBundle:
    model: object
    expanded_columns: Optional[List[str]] = None


class TransferForestPython:
    """Pure-Python transfer random forest implementation.

    This mirrors the original transfer forest style by training:
    - m_source: RF on source data
    - m0: RF on target data
    - m1: variable-importance weighted target RF
    - m2: source-error correction RF
    - m3: target RF with source prediction features appended
    and combining them with an ensemble rule.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        random_state: int = 42,
        min_samples_leaf: int = 1,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)
        self.min_samples_leaf = int(min_samples_leaf)

        self._is_classification = False
        self.classes_: Optional[np.ndarray] = None
        self.feature_names: List[str] = []

        self.models: Dict[str, object] = {}
        self.model_bundles: Dict[str, _ModelBundle] = {}
        self.ensemble_weights: Optional[np.ndarray] = None

    @staticmethod
    def _as_frame(X: pd.DataFrame | np.ndarray, columns: Optional[List[str]] = None) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if columns is not None and len(columns) == X.shape[1]:
            return pd.DataFrame(X, columns=columns)
        return pd.DataFrame(X)

    def _new_rf(self, classification: bool):
        if classification:
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                min_samples_leaf=self.min_samples_leaf,
                n_jobs=-1,
            )
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=-1,
        )

    def _fit_rf(self, X: pd.DataFrame, y: np.ndarray, classification: bool):
        model = self._new_rf(classification)
        model.fit(X, y)
        return model

    def _source_predict(self, X: pd.DataFrame) -> np.ndarray:
        m_source = self.models["m_source"]
        if self._is_classification:
            proba = m_source.predict_proba(X)
            return np.asarray(proba, dtype=float)
        return np.asarray(m_source.predict(X), dtype=float)

    def _importance_expanded_cols(self, X: pd.DataFrame, importances: np.ndarray) -> List[str]:
        imp = np.asarray(importances, dtype=float)
        if np.allclose(imp.sum(), 0.0):
            imp = np.ones_like(imp)
        norm = imp / imp.sum()
        # Use a small bounded expansion to emulate variable-importance weighting.
        repeats = np.clip(np.round(norm * 12).astype(int), 1, 6)

        expanded = []
        for col, rep in zip(X.columns, repeats):
            expanded.extend([col] * int(rep))
        return expanded

    def _m1_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        bundle = self.model_bundles.get("m1")
        if bundle is None or bundle.expanded_columns is None:
            return X
        return X.loc[:, bundle.expanded_columns]

    def _append_source_features(self, X: pd.DataFrame, source_pred: np.ndarray) -> pd.DataFrame:
        out = X.copy()
        if self._is_classification:
            for idx, cls in enumerate(self.classes_):
                out[f"y_source_hat_{cls}"] = source_pred[:, idx]
        else:
            out["y_source_hat"] = source_pred
        return out

    def fit(
        self,
        source_X: pd.DataFrame,
        source_y: np.ndarray,
        target_X: pd.DataFrame,
        target_y: np.ndarray,
    ) -> "TransferForestPython":
        source_X = source_X.copy()
        target_X = target_X.copy()
        self.feature_names = list(target_X.columns)

        unique_vals = np.unique(target_y)
        self._is_classification = np.array_equal(unique_vals, unique_vals.astype(int)) and unique_vals.size <= 6
        if self._is_classification:
            self.classes_ = np.unique(target_y)

        self.models["m_source"] = self._fit_rf(source_X, source_y, self._is_classification)

        m0 = self._fit_rf(target_X, target_y, self._is_classification)
        self.models["m0"] = m0

        source_importance = getattr(self.models["m_source"], "feature_importances_", np.ones(target_X.shape[1]))
        expanded_cols = self._importance_expanded_cols(target_X, np.asarray(source_importance))
        m1 = self._fit_rf(target_X.loc[:, expanded_cols], target_y, self._is_classification)
        self.models["m1"] = m1
        self.model_bundles["m1"] = _ModelBundle(model=m1, expanded_columns=expanded_cols)

        source_pred_target = self._source_predict(target_X)
        if self._is_classification:
            m2_models = []
            y_as_int = np.asarray(target_y)
            for class_idx, cls in enumerate(self.classes_):
                y_cls = (y_as_int == cls).astype(float)
                delta = y_cls - source_pred_target[:, class_idx]
                delta_model = self._fit_rf(target_X, delta, classification=False)
                m2_models.append(delta_model)
            self.models["m2"] = m2_models
        else:
            delta = np.asarray(target_y, dtype=float) - source_pred_target
            self.models["m2"] = self._fit_rf(target_X, delta, classification=False)

        m3_X = self._append_source_features(target_X, source_pred_target)
        m3 = self._fit_rf(m3_X, target_y, self._is_classification)
        self.models["m3"] = m3

        return self

    def _predict_m0(self, X: pd.DataFrame):
        if self._is_classification:
            return np.asarray(self.models["m0"].predict_proba(X), dtype=float)
        return np.asarray(self.models["m0"].predict(X), dtype=float)

    def _predict_m1(self, X: pd.DataFrame):
        X1 = self._m1_transform(X)
        if self._is_classification:
            return np.asarray(self.models["m1"].predict_proba(X1), dtype=float)
        return np.asarray(self.models["m1"].predict(X1), dtype=float)

    def _predict_m2(self, X: pd.DataFrame, pred_source: np.ndarray):
        if self._is_classification:
            corrected = np.zeros_like(pred_source, dtype=float)
            for idx, delta_model in enumerate(self.models["m2"]):
                delta = np.asarray(delta_model.predict(X), dtype=float)
                corrected[:, idx] = pred_source[:, idx] + delta
            corrected = np.clip(corrected, 0.0, None)
            denom = corrected.sum(axis=1, keepdims=True)
            denom[denom == 0] = 1.0
            return corrected / denom
        delta = np.asarray(self.models["m2"].predict(X), dtype=float)
        return pred_source + delta

    def _predict_m3(self, X: pd.DataFrame, pred_source: np.ndarray):
        X3 = self._append_source_features(X, pred_source)
        if self._is_classification:
            return np.asarray(self.models["m3"].predict_proba(X3), dtype=float)
        return np.asarray(self.models["m3"].predict(X3), dtype=float)

    def _predict_ensemble(
        self,
        pred_0,
        pred_1,
        pred_2,
        pred_3,
        x_ensemble: Optional[pd.DataFrame] = None,
        y_ensemble: Optional[np.ndarray] = None,
    ):
        if self._is_classification:
            if x_ensemble is None or y_ensemble is None:
                combined = pred_0 * pred_1 * pred_2 * pred_3
                denom = combined.sum(axis=1, keepdims=True)
                denom[denom == 0] = 1.0
                return combined / denom

            src_ens = self._source_predict(x_ensemble)
            ens_0 = self._predict_m0(x_ensemble)
            ens_1 = self._predict_m1(x_ensemble)
            ens_2 = self._predict_m2(x_ensemble, src_ens)
            ens_3 = self._predict_m3(x_ensemble, src_ens)

            meta_X = np.hstack([ens_0, ens_1, ens_2, ens_3])
            meta = RandomForestClassifier(
                n_estimators=max(200, self.n_estimators // 2),
                random_state=self.random_state,
                n_jobs=-1,
            )
            meta.fit(meta_X, y_ensemble)
            imp = np.asarray(meta.feature_importances_, dtype=float)
            n_classes = len(self.classes_)
            weight_matrix = imp.reshape(4, n_classes).T
            weight_matrix = weight_matrix / np.clip(weight_matrix.sum(axis=1, keepdims=True), 1e-12, None)

            preds = [pred_0, pred_1, pred_2, pred_3]
            out = np.ones_like(pred_0, dtype=float)
            for cls_idx in range(n_classes):
                for model_idx in range(4):
                    out[:, cls_idx] *= np.power(
                        np.clip(preds[model_idx][:, cls_idx], 1e-12, 1.0),
                        weight_matrix[cls_idx, model_idx],
                    )
            denom = out.sum(axis=1, keepdims=True)
            denom[denom == 0] = 1.0
            return out / denom

        if x_ensemble is None or y_ensemble is None:
            return (pred_0 + pred_1 + pred_2 + pred_3) / 4.0

        src_ens = self._source_predict(x_ensemble)
        ens_0 = self._predict_m0(x_ensemble)
        ens_1 = self._predict_m1(x_ensemble)
        ens_2 = self._predict_m2(x_ensemble, src_ens)
        ens_3 = self._predict_m3(x_ensemble, src_ens)

        meta_X = np.column_stack([ens_0, ens_1, ens_2, ens_3])
        meta = RandomForestRegressor(
            n_estimators=max(200, self.n_estimators // 2),
            random_state=self.random_state,
            n_jobs=-1,
        )
        meta.fit(meta_X, y_ensemble)
        weights = np.asarray(meta.feature_importances_, dtype=float)
        weights = weights / np.clip(weights.sum(), 1e-12, None)
        self.ensemble_weights = weights
        return (pred_0 * weights[0]) + (pred_1 * weights[1]) + (pred_2 * weights[2]) + (pred_3 * weights[3])

    def predict_trans_rf(
        self,
        newdata: pd.DataFrame,
        x_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        x_ensemble: Optional[pd.DataFrame] = None,
        y_ensemble: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        X = newdata.copy()
        pred_source = self._source_predict(X)
        pred_0 = self._predict_m0(X)
        pred_1 = self._predict_m1(X)
        pred_2 = self._predict_m2(X, pred_source)
        pred_3 = self._predict_m3(X, pred_source)
        pred_ensemble = self._predict_ensemble(pred_0, pred_1, pred_2, pred_3, x_ensemble, y_ensemble)

        out: Dict[str, np.ndarray] = {}
        if self._is_classification:
            pred_source_class = np.asarray(self.classes_)[np.argmax(pred_source, axis=1)]
            pred_0_class = np.asarray(self.classes_)[np.argmax(pred_0, axis=1)]
            pred_1_class = np.asarray(self.classes_)[np.argmax(pred_1, axis=1)]
            pred_2_class = np.asarray(self.classes_)[np.argmax(pred_2, axis=1)]
            pred_3_class = np.asarray(self.classes_)[np.argmax(pred_3, axis=1)]
            pred_ensemble_class = np.asarray(self.classes_)[np.argmax(pred_ensemble, axis=1)]

            out["pred_source"] = pred_source_class
            out["pred_0"] = pred_0_class
            out["pred_1"] = pred_1_class
            out["pred_2"] = pred_2_class
            out["pred_3"] = pred_3_class
            out["pred_ensemble"] = pred_ensemble_class
            for idx in range(pred_source.shape[1]):
                pidx = idx + 1
                out[f"pred_source_prob_{pidx}"] = pred_source[:, idx]
                out[f"pred_0_prob_{pidx}"] = pred_0[:, idx]
                out[f"pred_1_prob_{pidx}"] = pred_1[:, idx]
                out[f"pred_2_prob_{pidx}"] = pred_2[:, idx]
                out[f"pred_3_prob_{pidx}"] = pred_3[:, idx]
                out[f"pred_ensemble_prob_{pidx}"] = pred_ensemble[:, idx]
        else:
            out["pred_source"] = pred_source
            out["pred_0"] = pred_0
            out["pred_1"] = pred_1
            out["pred_2"] = pred_2
            out["pred_3"] = pred_3
            out["pred_ensemble"] = pred_ensemble

        if x_val is not None:
            val = self.predict_trans_rf(
                x_val,
                x_val=None,
                y_val=None,
                x_ensemble=x_ensemble,
                y_ensemble=y_ensemble,
            )
            for key, value in val.items():
                out[f"{key}_val"] = value

        if y_val is not None:
            out["truth"] = np.asarray(y_val)

        return out

    def generate_predictions(
        self,
        views: List[pd.DataFrame],
        response: Optional[np.ndarray] = None,
        validation_views: Optional[List[pd.DataFrame]] = None,
        validation_response: Optional[np.ndarray] = None,
        ensemble_views: Optional[List[pd.DataFrame]] = None,
        ensemble_response: Optional[np.ndarray] = None,
        integration_type=None,
    ) -> List[Dict[str, np.ndarray]]:
        _ = integration_type
        results = []
        for idx, view in enumerate(views):
            x_val = None if validation_views is None else validation_views[idx]
            x_ensemble = None if ensemble_views is None else ensemble_views[idx]
            y_val = validation_response if validation_response is not None else response
            res = self.predict_trans_rf(
                newdata=view,
                x_val=x_val,
                y_val=y_val,
                x_ensemble=x_ensemble,
                y_ensemble=ensemble_response,
            )
            results.append(res)
        return results
