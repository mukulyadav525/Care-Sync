"""Pooled multi-dataset training: one global HR/HRV forecaster trained on 70%
of subjects pooled across Dataset3 (Stress-Predict) + WESAD, tested on the
held-out 30%.

Both datasets are Empatica E4 exports sharing the same six vitals
(HR, IBI, ACC, BVP, EDA, TEMP), so the pooled model uses the full feature set.

Key design choices
------------------
* SUBJECT-LEVEL split (not window-level): 70% of subjects -> train, 30% -> test,
  stratified per dataset so each dataset appears in both. This holds out WHOLE
  PEOPLE, which is the honest test of "does it generalise to a new person?"
  (window-level random splitting would leak a subject into both sides).
* Feature columns are INTERSECTED across subjects so the global model only uses
  features every subject actually has.
* Memory safety: windows per subject are capped/subsampled, controlled by
  cfg["multi"]["stride_s"] and cfg["multi"]["max_windows_per_subject"].

Reports overall test metrics plus a per-dataset breakdown.
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import joblib
import numpy as np

from ..evaluation import metrics as eval_metrics
from ..features import build_features
from ..models import datasets
from .train_pipeline import load_and_featurize


def discover_subjects(fmt: str, raw_root: Path) -> list[str]:
    """E4 (Stress-Predict / WESAD): subject folders starting with 'S'."""
    raw_root = Path(raw_root)
    return sorted(p.name for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("S"))


def _dataset_cfg(shared: dict, ds: dict, project_root: Path) -> tuple[dict, Path]:
    """Build a per-dataset cfg (shared model/cleaning + that dataset's data block)."""
    cfg = copy.deepcopy(shared)
    cfg["data"] = {"format": ds["format"], "raw_root": ds["raw_root"], "subjects": ds.get("subjects")}
    raw_root = (project_root.parent / ds["raw_root"]).resolve()
    return cfg, raw_root


def _subsample(X: np.ndarray, y: np.ndarray, cap: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    if cap and len(X) > cap:
        idx = sorted(rng.sample(range(len(X)), cap))
        return X[idx], y[idx]
    return X, y


def run_multi_dataset(cfg: dict, project_root: Path | str, out_dir: Path | str,
                      model_type: str = "tcn", holdout_dataset: str | None = None) -> dict:
    project_root = Path(project_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    horizons = cfg["model"]["horizons_s"]
    seq_len = 1 if model_type == "xgboost" else cfg["model"]["input_seq_len_s"]
    mcfg = cfg.get("multi", {})
    stride = mcfg.get("stride_s", max(cfg["model"]["stride_s"], 15))
    cap = mcfg.get("max_windows_per_subject", 4000)
    split_cfg = cfg.get("split", {})
    train_frac = split_cfg.get("train_fraction", 0.6)
    val_frac = split_cfg.get("val_fraction", 0.15)
    seed = split_cfg.get("seed", 42)
    rng = random.Random(seed)

    # ---- 1) assign subjects to train/val/test per dataset (subject-level holdout) ----
    # 3-way split so hyper-params / thresholds are tuned on VAL and the TEST set
    # is touched only once. Stratified per dataset so each appears in all folds.
    # If `holdout_dataset` is set, ALL of that dataset's subjects become test and
    # every other dataset is split train/val only (leave-one-dataset-out, #15).
    roster = []  # (dataset_name, fmt, raw_root, subject, role)
    for ds in cfg["datasets"]:
        ds_cfg, raw_root = _dataset_cfg(cfg, ds, project_root)
        subs = ds.get("subjects") or discover_subjects(ds["format"], raw_root)
        subs = list(subs)
        rng.shuffle(subs)
        n = len(subs)
        if holdout_dataset is not None:
            if ds["name"] == holdout_dataset:
                roles = ["test"] * n
            else:
                n_val = max(1, int(round(n * val_frac))) if n >= 2 else 0
                roles = ["val"] * n_val + ["train"] * (n - n_val)
        else:
            n_train = int(round(n * train_frac))
            n_val = int(round(n * val_frac))
            if n >= 3:
                n_train = min(n_train, n - 2)
                n_val = max(1, min(n_val, n - n_train - 1))
            roles = ["train" if i < n_train else ("val" if i < n_train + n_val else "test")
                     for i in range(n)]
        for s, role in zip(subs, roles):
            roster.append((ds["name"], ds["format"], raw_root, s, role))

    if not roster:
        raise ValueError("No subjects discovered across the configured datasets.")

    # ---- 2) featurize + window every subject ----
    target_cols = None  # resolved from the first subject's table via build_target_cols

    # stable integer id per dataset, used as a domain-adaptation feature (#15)
    dataset_ids = {ds["name"]: i for i, ds in enumerate(cfg["datasets"])}
    add_dataset_id = cfg.get("multi", {}).get("dataset_id_feature", True)

    per_subject = []  # (role, dataset, sid, X, y, cols)
    feature_cols = None
    for ds_name, fmt, raw_root, sid, role in roster:
        ds_cfg = copy.deepcopy(cfg)
        ds_cfg["data"] = {"format": fmt, "raw_root": str(raw_root)}
        try:
            table, _ = load_and_featurize(raw_root / sid, ds_cfg)
        except Exception as exc:  # noqa: BLE001 - skip unreadable subjects, keep going
            print(f"[multi] skip {ds_name}/{sid}: {exc}")
            continue

        if add_dataset_id:
            table["dataset_id"] = float(dataset_ids[ds_name])
        cols = build_features.numeric_feature_columns(table, cfg)
        tcols = build_features.build_target_cols(table, cfg)
        cols = [c for c in cols if c not in tcols]
        if not tcols:
            print(f"[multi] skip {ds_name}/{sid}: no targets")
            continue

        X, y, _ = datasets.make_windows(table, cols, tcols, seq_len, stride)
        if len(X) < 20:
            print(f"[multi] skip {ds_name}/{sid}: only {len(X)} windows")
            continue
        X, y = _subsample(X, y, cap, rng)

        feature_cols = cols if feature_cols is None else [c for c in feature_cols if c in set(cols)]
        per_subject.append((role, ds_name, sid, X, y, cols, tcols))

    if feature_cols is None or not per_subject:
        raise ValueError("No usable subjects after featurization.")

    # consistent target set across subjects (intersection)
    target_cols = list(per_subject[0][6])
    for *_, tcols in per_subject:
        target_cols = [t for t in target_cols if t in tcols]

    def align(X, cols):
        idx = [cols.index(c) for c in feature_cols]
        return X[:, :, idx]

    def t_align(y, tcols):
        idx = [tcols.index(t) for t in target_cols]
        return y[:, idx]

    def collect(role):
        return [(ds, sid, align(X, cols), t_align(y, tc))
                for r, ds, sid, X, y, cols, tc in per_subject if r == role]

    train, val, test = collect("train"), collect("val"), collect("test")
    if not train or not test:
        raise ValueError(f"Need subjects in train and test (train={len(train)}, test={len(test)}). "
                         "Add more subjects or adjust split fractions.")

    X_train = np.concatenate([x for _, _, x, _ in train], axis=0)
    y_train = np.concatenate([y for _, _, _, y in train], axis=0)

    scaler = datasets.fit_scaler(X_train)
    X_train_s = datasets.apply_scaler(X_train, scaler)

    # subject-level validation windows (for early stopping AND threshold calibration)
    X_val_s = y_val = None
    if val:
        X_val = np.concatenate([x for _, _, x, _ in val], axis=0)
        y_val = np.concatenate([y for _, _, _, y in val], axis=0)
        X_val_s = datasets.apply_scaler(X_val, scaler)

    summary = {
        "n_train_subjects": len(train), "n_val_subjects": len(val), "n_test_subjects": len(test),
        "n_train_windows": len(X_train), "feature_cols": feature_cols,
        "target_cols": target_cols, "model_type": model_type, "seed": seed,
        "datasets": sorted({ds for ds, *_ in train} | {ds for ds, *_ in test}),
    }

    # ---- reproducible split metadata ----
    split_meta = {
        "seed": seed, "train_fraction": train_frac, "val_fraction": val_frac,
        "train": [f"{ds}/{sid}" for ds, sid, _, _ in train],
        "val": [f"{ds}/{sid}" for ds, sid, _, _ in val],
        "test": [f"{ds}/{sid}" for ds, sid, _, _ in test],
    }
    (out_dir / "split.json").write_text(json.dumps(split_meta, indent=2))

    # ---- 3) train ----
    if model_type == "xgboost":
        _train_xgb(X_train_s, y_train, target_cols, out_dir)
        predict = lambda Xs: _predict_xgb(out_dir, Xs)  # noqa: E731
    else:
        predict = _train_deep(X_train_s, y_train, X_val_s, y_val, scaler,
                              target_cols, feature_cols, cfg, model_type, out_dir)

    joblib.dump(scaler, out_dir / "scaler.joblib")
    joblib.dump({**summary, "seq_len": seq_len}, out_dir / "meta.joblib")

    # ---- 4) evaluate: overall + per-dataset + per-subject + PI coverage ----
    def eval_split(subjects):
        per_dataset_pred, subj_results = {}, []
        all_t, all_m, all_s = [], [], []
        for ds, sid, X, y in subjects:
            Xs = datasets.apply_scaler(X, scaler)
            mean, std = predict(Xs)
            all_t.append(y); all_m.append(mean); all_s.append(std)
            subj_results.append((f"{ds}/{sid}", y, mean))
            per_dataset_pred.setdefault(ds, [[], [], []])
            per_dataset_pred[ds][0].append(y); per_dataset_pred[ds][1].append(mean); per_dataset_pred[ds][2].append(std)
        yt, ym, ys = np.concatenate(all_t), np.concatenate(all_m), np.concatenate(all_s)
        return {
            "forecast_report": eval_metrics.forecast_report(yt, ym, ys, target_cols),
            "per_dataset": {ds: eval_metrics.forecast_report(np.concatenate(t), np.concatenate(m), np.concatenate(s), target_cols)
                            for ds, (t, m, s) in per_dataset_pred.items()},
            "per_subject": eval_metrics.per_subject_metrics(subj_results, target_cols),
        }, (yt, ym, ys)

    test_eval, _ = eval_split(test)
    val_eval = eval_split(val)[0] if val else None

    return {"summary": summary, "split": split_meta, "test": test_eval, "val": val_eval,
            "train_subjects": [(ds, sid) for ds, sid, _, _ in train],
            "val_subjects": [(ds, sid) for ds, sid, _, _ in val],
            "test_subjects": [(ds, sid) for ds, sid, _, _ in test]}


def _train_deep(X_train_s, y_train, X_val_s, y_val, scaler, target_cols, feature_cols, cfg, model_type, out_dir):
    import torch  # noqa: PLC0415
    from ..models import train as train_mod  # noqa: PLC0415
    from ..models.torch_datasets import SequenceDataset  # noqa: PLC0415

    target_scaler = datasets.fit_target_scaler(y_train)
    y_train_s = datasets.apply_target_scaler(y_train, target_scaler)
    # Prefer the subject-level validation set for early stopping; fall back to a
    # chronological slice of train only if no val subjects were provided.
    if X_val_s is not None and len(X_val_s) > 0:
        train_ds = SequenceDataset(X_train_s, y_train_s)
        val_ds = SequenceDataset(X_val_s, datasets.apply_target_scaler(y_val, target_scaler))
    else:
        tr, va, _ = datasets.chronological_split(len(X_train_s), 0.1, 0.0)
        train_ds = SequenceDataset(X_train_s[tr], y_train_s[tr])
        val_ds = SequenceDataset(X_train_s[va], y_train_s[va])

    n_features, n_horizons = len(feature_cols), len(target_cols)
    if model_type == "tcn":
        from ..models.tcn import TCNForecaster  # noqa: PLC0415
        model = TCNForecaster(n_features=n_features, n_horizons=n_horizons,
                              hidden_channels=cfg["model"]["hidden_channels"], levels=cfg["model"]["tcn_levels"],
                              kernel_size=cfg["model"]["kernel_size"], dropout=cfg["model"]["dropout"])
    elif model_type in ("lstm", "gru"):
        from ..models.lstm_gru import GRUForecaster, LSTMForecaster  # noqa: PLC0415
        ctor = LSTMForecaster if model_type == "lstm" else GRUForecaster
        model = ctor(n_features=n_features, n_horizons=n_horizons, hidden_size=cfg["model"]["hidden_channels"])
    elif model_type == "transformer":
        from ..models.transformer import TransformerForecaster  # noqa: PLC0415
        model = TransformerForecaster(n_features=n_features, n_horizons=n_horizons, d_model=cfg["model"]["hidden_channels"])
    else:
        raise ValueError(model_type)

    train_mod.train_model(model, train_ds, val_ds, epochs=cfg["model"]["epochs"],
                          lr=cfg["model"]["lr"], batch_size=cfg["model"]["batch_size"])
    torch.save(model.state_dict(), out_dir / f"{model_type}_model.pt")
    joblib.dump(target_scaler, out_dir / "target_scaler.joblib")

    def predict(Xs):
        mean_s, std_s = train_mod.predict(model, Xs)
        mean = datasets.inverse_target_scaler(mean_s, target_scaler)
        std = datasets.inverse_target_std(std_s, target_scaler)
        return mean, std
    return predict


def _train_xgb(X_train_s, y_train, target_cols, out_dir):
    from ..models.xgb_model import XGBForecaster  # noqa: PLC0415
    model = XGBForecaster(n_targets=len(target_cols))
    model.fit(X_train_s[:, 0, :], y_train)
    model.save(out_dir / "xgb")


def _predict_xgb(out_dir, Xs):
    from ..models.xgb_model import XGBForecaster  # noqa: PLC0415
    import joblib as _jl  # noqa: PLC0415
    meta = _jl.load(out_dir / "meta.joblib")
    model = XGBForecaster(n_targets=len(meta["target_cols"]))
    model.load(out_dir / "xgb")
    mean, std = model.predict(Xs[:, 0, :])
    return mean, std


def run_leave_one_dataset_out(cfg: dict, project_root: Path | str, out_dir: Path | str,
                              model_type: str = "tcn") -> dict:
    """Train on all datasets but one, test on the held-out dataset — repeated for
    each dataset. Measures cross-dataset generalisation / domain shift (#15)."""
    out_dir = Path(out_dir)
    results = {}
    for ds in cfg["datasets"]:
        name = ds["name"]
        res = run_multi_dataset(cfg, project_root, out_dir / f"holdout_{name}",
                                model_type=model_type, holdout_dataset=name)
        results[name] = res["test"]["forecast_report"]["overall"]
    return results


def run_repeated_splits(cfg: dict, project_root: Path | str, out_dir: Path | str,
                        model_type: str = "tcn", n_repeats: int = 5) -> dict:
    """Leave-subjects-out style robustness check: re-run the pooled train/val/test
    with a different random subject split each time (seed + i) and aggregate the
    overall test metrics as mean +/- std across repeats."""
    out_dir = Path(out_dir)
    base_seed = cfg.get("split", {}).get("seed", 42)
    folds = []
    for i in range(n_repeats):
        cfg_i = copy.deepcopy(cfg)
        cfg_i.setdefault("split", {})["seed"] = base_seed + i
        res = run_multi_dataset(cfg_i, project_root, out_dir / f"fold{i}", model_type=model_type)
        folds.append(res["test"]["forecast_report"]["overall"])

    agg = {}
    for key in ("MAE", "RMSE", "R2", "pearson_r", "bias"):
        vals = np.array([f[key] for f in folds if np.isfinite(f[key])], float)
        if len(vals):
            agg[key] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return {"n_repeats": n_repeats, "aggregate": agg, "folds": folds}
