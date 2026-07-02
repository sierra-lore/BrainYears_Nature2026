#!/usr/bin/env python3
"""Regenerate BrainYears SVG figure panels."""

import re
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
def display(*args, **kwargs):
    return None

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from scipy import stats

import warnings

sns.set_theme(style="ticks", context="paper")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")
np.seterr(over="ignore", invalid="ignore")

PALETTE = {
    "ink": "#1A1A1A",
    "slate": "#4B5563",
    "blue": "#2F5D8C",
    "teal": "#2A7F7F",
    "green": "#5A8F5A",
    "orange": "#D07C4E",
    "gold": "#C7A44A",
    "red": "#B2493E",
    "purple": "#6C5B7B",
    "gray": "#6E6E6E",
    "light_gray": "#D9D9D9",
    "white": "#FFFFFF",
}

CATEGORY_PALETTE = {
    "ERP Inhibit": "#B2493E",
    "ERP Undistracted": "#D07C4E",
    "ERP Distracted": "#C7A44A",
    "Gamma": "#6C5B7B",
    "Delta": "#2F5D8C",
    "Theta": "#2A7F7F",
    "Alpha": "#5A8F5A",
    "Beta": "#4B5563",
    "Behavioral": "#1A1A1A",
    "Composite": "#6E6E6E",
    "Demographics": "#4C72B0",
    "Spectral": "#8DAA91",
}

CONTROL_COLOR = PALETTE["blue"]
INTERVENTION_COLOR = PALETTE["orange"]
AGE_CMAP = "cividis"

plt.rcParams.update({
    "figure.dpi": 480,
    "savefig.dpi": 1200,
    "svg.fonttype": "none",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": PALETTE["slate"],
    "axes.grid": False,
    "xtick.bottom": True,
    "ytick.left": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
})

PANEL_FIGSIZE = (5.5, 5)

def style_axis(ax):
    ax.set_facecolor(PALETTE["white"])
    ax.grid(False)
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    ax.tick_params(
        axis="both", which="both", bottom=True, left=True, top=False, right=False,
        length=5, width=1.0, direction="out",
        color=PALETTE["slate"], labelcolor=PALETTE["ink"],
    )
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")


def panel_label(ax, label, x=-0.08, y=1.04):
    ax.text(
        x, y, label, transform=ax.transAxes,
        fontsize=12, fontweight="bold",
        va="top", ha="left", color=PALETTE["ink"],
    )


def _new_panel():
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    return fig, ax


# %%
BASE = Path(__file__).resolve().parent
DATASETS = {"model": pd.read_csv(BASE / "model.csv")}

GROUPING_RULES = [
    ("ERP Inhibit", ["INHIBIT", "CZ_Pe", "CZ_ERN", "NOGO-INCONGRUENT", "NOGO-CONGRUENT"]),
    ("ERP Undistracted", ["_UNDISTRACTED", "FLANKER-CONGRUENT"]),
    ("ERP Distracted", ["_DISTRACTED", "FLANKER-INCONGRUENT"]),
    ("Gamma", ["gamma"]),
    ("Delta", ["delta"]),
    ("Theta", ["theta"]),
    ("Alpha", ["alpha", "iaf"]),
    ("Beta", ["beta"]),
    ("Behavioral", ["RT-", "AC-", "AC_", "RT_"]),
    ("Composite", ["c_"]),
    ("Demographics", ["race", "sex", "education"]),
]


def categorize_primary(name):
    n = name.lower()
    for group, tokens in GROUPING_RULES:
        if any(token.lower() in n for token in tokens):
            return group
    return "Spectral"


# Prepare model matrix (aligned with model.py)

df_model_raw = DATASETS["model"].copy()
df_model_raw["age"] = pd.to_numeric(df_model_raw.get("age"), errors="coerce")

age_mask = df_model_raw["age"].between(18.0, 89.99)
df_model = df_model_raw.loc[age_mask].copy()

X_all_cols = df_model.drop(columns=["age"])
not_nan_mask = ~X_all_cols.isna().any(axis=1)
df_model_filtered = df_model.loc[not_nan_mask].copy()

X_model_all = df_model_filtered.select_dtypes(include=[np.number]).drop(columns=["age"], errors="ignore")
y_model = pd.to_numeric(df_model_filtered["age"], errors="coerce")


def load_model_constants(path):
    constants = {}
    try:
        tree = ast.parse(Path(path).read_text())
    except FileNotFoundError:
        return constants
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if not name.isupper():
                continue
            try:
                constants[name] = ast.literal_eval(node.value)
            except Exception:
                continue
    return constants


def load_sorted_bin_assignments(path):
    try:
        text = Path(path).read_text()
    except FileNotFoundError:
        return None
    match = re.search(r"SORTED_BIN_ASSIGNMENTS\s*=\s*np\.array\((\[[\s\S]*?\])\)", text)
    if not match:
        return None
    try:
        values = ast.literal_eval(match.group(1))
    except Exception:
        return None
    return np.asarray(values, dtype=int)


def get_const(name, default):
    return MODEL_CONSTANTS.get(name, default)


MODEL_CONSTANTS = load_model_constants(BASE / "model.py")

RANDOM_STATE = get_const("RANDOM_STATE", 52)
STRATIFY_BINS = get_const("STRATIFY_BINS", 18)
CLIP_PERCENTILE = get_const("CLIP_PERCENTILE", 0.5)
VARIANCE_THRESHOLD = get_const("VARIANCE_THRESHOLD", 1e-7)

ELASTICNET_ALPHA = get_const("ELASTICNET_ALPHA", 0.325)
ELASTICNET_L1_RATIO = get_const("ELASTICNET_L1_RATIO", 0.235)
ELASTICNET_SELECTION = get_const("ELASTICNET_SELECTION", "random")

GBR_N_ESTIMATORS = get_const("GBR_N_ESTIMATORS", 185)
GBR_LEARNING_RATE = get_const("GBR_LEARNING_RATE", 0.0175)
GBR_MAX_DEPTH = get_const("GBR_MAX_DEPTH", 2)
GBR_SUBSAMPLE = get_const("GBR_SUBSAMPLE", 0.5)
GBR_MIN_SAMPLES_LEAF = get_const("GBR_MIN_SAMPLES_LEAF", 60)
GBR_LOSS = get_const("GBR_LOSS", "huber")

BIAS_POLY_DEGREE = get_const("BIAS_POLY_DEGREE", 3)
SORTED_BIN_ASSIGNMENTS = load_sorted_bin_assignments(BASE / "model.py")


def fit_winsor_clip_bounds(train_features, clip_percentile):
    lo = np.percentile(train_features, clip_percentile, axis=0)
    hi = np.percentile(train_features, 100 - clip_percentile, axis=0)
    return lo, hi


def apply_winsor_clipping(features, lo, hi):
    return np.clip(features, lo, hi)


def fit_bundle(X_train, y_train, feature_names, use_gbr=True):
    vt = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    X_train_vt = vt.fit_transform(X_train)

    clip_lo, clip_hi = fit_winsor_clip_bounds(X_train_vt, CLIP_PERCENTILE)
    X_train_clip = apply_winsor_clipping(X_train_vt, clip_lo, clip_hi)

    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    X_train_pt = pt.fit_transform(X_train_clip)
    ss = StandardScaler()
    X_train_scaled = ss.fit_transform(X_train_pt)

    enet = ElasticNet(
        alpha=ELASTICNET_ALPHA,
        l1_ratio=ELASTICNET_L1_RATIO,
        selection=ELASTICNET_SELECTION,
        random_state=RANDOM_STATE,
        max_iter=100000,
        tol=1e-5,
    ).fit(X_train_scaled, y_train)

    enet_pred = enet.predict(X_train_scaled)

    if use_gbr:
        residuals = y_train - enet_pred
        X_gbr = np.column_stack([X_train_clip, enet_pred])
        gbr = GradientBoostingRegressor(
            n_estimators=GBR_N_ESTIMATORS,
            learning_rate=GBR_LEARNING_RATE,
            max_depth=GBR_MAX_DEPTH,
            subsample=GBR_SUBSAMPLE,
            min_samples_leaf=GBR_MIN_SAMPLES_LEAF,
            loss=GBR_LOSS,
            random_state=RANDOM_STATE,
        ).fit(X_gbr, residuals)
        pred_raw = enet_pred + gbr.predict(X_gbr)
    else:
        gbr = None
        pred_raw = enet_pred

    poly = PolynomialFeatures(degree=BIAS_POLY_DEGREE, include_bias=False)
    bias = LinearRegression().fit(
        poly.fit_transform(pred_raw.reshape(-1, 1)), pred_raw - y_train
    )

    return {
        "feats": list(feature_names),
        "vt": vt,
        "clip_lo": clip_lo,
        "clip_hi": clip_hi,
        "pt": pt,
        "ss": ss,
        "enet": enet,
        "gbr": gbr,
        "poly": poly,
        "bias": bias,
    }


def transform_features(bundle, X):
    X_vt = bundle["vt"].transform(X)
    X_clip = np.clip(X_vt, bundle["clip_lo"], bundle["clip_hi"])
    X_scaled = bundle["ss"].transform(bundle["pt"].transform(X_clip))
    return X_clip, X_scaled


def predict_bundle(bundle, X):
    X_clip, X_scaled = transform_features(bundle, X)
    enet_pred = bundle["enet"].predict(X_scaled)
    if bundle.get("gbr") is not None:
        X_gbr = np.column_stack([X_clip, enet_pred])
        pred_raw = enet_pred + bundle["gbr"].predict(X_gbr)
    else:
        pred_raw = enet_pred
    pred = pred_raw - bundle["bias"].predict(
        bundle["poly"].transform(pred_raw.reshape(-1, 1))
    )
    return pred, pred_raw


def to_matrix(df, features):
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing {len(missing)} feature columns. Example: {missing[:10]}")
    X = df[features].apply(pd.to_numeric, errors="coerce")
    return X.to_numpy(dtype=float)


def load_model_bundle(path):
    try:
        return joblib.load(path)
    except Exception as exc:
        print(
            f"Warning: could not load model bundle ({type(exc).__name__}: {exc}). "
            "Falling back to refit."
        )
        return None


def make_stratify_bins(y, n_bins, sorted_bin_assignments=None):
    if sorted_bin_assignments is not None:
        try:
            if len(y) != len(sorted_bin_assignments):
                raise ValueError(
                    f"Sample count mismatch: expected {len(sorted_bin_assignments)}, got {len(y)}"
                )
            sort_idx = np.argsort(y)
            age_bins = np.empty(len(y), dtype=int)
            age_bins[sort_idx] = sorted_bin_assignments
            _, bin_counts = np.unique(age_bins, return_counts=True)
            if np.all(bin_counts >= 2):
                return age_bins
        except Exception as exc:
            print(f"Warning: model.py stratify binning failed ({type(exc).__name__}: {exc}).")
    if not n_bins or n_bins < 2:
        return None
    try:
        bins = pd.qcut(pd.Series(y), q=n_bins, duplicates="drop")
        if (bins.value_counts() >= 2).all():
            return bins
    except Exception:
        return None
    return None


bundle_full = load_model_bundle(BASE / "model_bundle.joblib")

if bundle_full is None:
    feature_names = X_model_all.columns.to_list()
    X_model = X_model_all.copy()
else:
    feature_names = list(bundle_full.get("feats", X_model_all.columns))
    missing = [c for c in feature_names if c not in X_model_all.columns]
    if missing:
        raise KeyError(
            f"Missing {len(missing)} feature columns in model.csv. Example: {missing[:10]}"
        )
    X_model = X_model_all.reindex(columns=feature_names)

scale = X_model.abs().max(skipna=True).replace(0, 1.0).fillna(1.0)
X_model_rescaled = X_model.divide(scale)

X_all = X_model.to_numpy(dtype=float)
y_all = y_model.to_numpy(float, copy=False)

TEST_SIZE = 0.2
idx_all = np.arange(len(y_all))
stratify_bins = make_stratify_bins(y_all, STRATIFY_BINS, SORTED_BIN_ASSIGNMENTS)
train_idx, test_idx = train_test_split(
    idx_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_bins
)
X_train, X_test = X_all[train_idx], X_all[test_idx]
y_train, y_test = y_all[train_idx], y_all[test_idx]

bundle_eval = bundle_full
if bundle_eval is None:
    bundle_eval = fit_bundle(X_train, y_train, feature_names, use_gbr=True)
pred_post, pred_pre = predict_bundle(bundle_eval, X_test)

residuals_post = pred_post - y_test

bin_size = 10
bin_start = np.floor(y_test.min() / bin_size) * bin_size
bin_end = np.ceil(y_test.max() / bin_size) * bin_size
bin_edges = np.arange(bin_start, bin_end + bin_size, bin_size)
if len(bin_edges) < 3:
    age_bins = pd.qcut(y_test, q=4, duplicates="drop")
else:
    age_bins = pd.cut(y_test, bins=bin_edges, include_lowest=True)

residuals_df = pd.DataFrame({
    "age": y_test,
    "residual": residuals_post,
    "age_bin": age_bins,
})
grouped_resid = residuals_df.groupby("age_bin", observed=True)["residual"]
mae_by_bin = (
    grouped_resid.apply(lambda s: float(np.mean(np.abs(s))))
    .reset_index()
    .rename(columns={"residual": "mae"})
)
mae_by_bin["count"] = grouped_resid.size().values
mae_by_bin["sem"] = grouped_resid.apply(
    lambda s: float(np.std(np.abs(s), ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0
).values
mae_by_bin["label"] = mae_by_bin["age_bin"].astype(str)


# %%
def compute_selection_frequency(X, y, feature_names, n_splits, test_size, random_state):
    counts = pd.Series(0, index=feature_names, dtype=float)
    idx_all = np.arange(len(y))
    for split_seed in range(n_splits):
        tr_idx, _ = train_test_split(
            idx_all, test_size=test_size, random_state=random_state + split_seed
        )
        X_tr = X[tr_idx]
        y_tr = y[tr_idx]

        vt = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
        X_tr_vt = vt.fit_transform(X_tr)
        vt_mask = vt.get_support()
        if not np.any(vt_mask):
            continue

        clip_lo, clip_hi = fit_winsor_clip_bounds(X_tr_vt, CLIP_PERCENTILE)
        X_tr_clip = apply_winsor_clipping(X_tr_vt, clip_lo, clip_hi)
        pt = PowerTransformer(method="yeo-johnson", standardize=True)
        X_tr_pt = pt.fit_transform(X_tr_clip)
        ss = StandardScaler()
        X_tr_scaled = ss.fit_transform(X_tr_pt)

        enet_model = ElasticNet(
            alpha=ELASTICNET_ALPHA,
            l1_ratio=ELASTICNET_L1_RATIO,
            selection=ELASTICNET_SELECTION,
            random_state=RANDOM_STATE,
            max_iter=100000,
            tol=1e-5,
        )
        enet_model.fit(X_tr_scaled, y_tr)
        selected_mask = np.zeros(len(feature_names), dtype=bool)
        selected_mask[np.where(vt_mask)[0]] = np.abs(enet_model.coef_) > 1e-6
        counts[selected_mask] += 1

    return counts / float(n_splits)


def compute_permutation_importance(
    bundle,
    X,
    y,
    feature_names,
    feature_list,
    n_repeats=10,
    seed=RANDOM_STATE,
):
    if bundle is None or len(feature_list) == 0:
        return pd.DataFrame(columns=["feature", "delta_mae", "sem"])
    feature_to_idx = {name: i for i, name in enumerate(feature_names)}
    X_base = np.asarray(X, dtype=float)
    baseline_pred, _ = predict_bundle(bundle, X_base)
    baseline_mae = mean_absolute_error(y, baseline_pred)
    rng = np.random.default_rng(seed)
    rows = []
    for feat in feature_list:
        idx = feature_to_idx.get(feat)
        if idx is None:
            continue
        deltas = []
        for _ in range(n_repeats):
            X_perm = X_base.copy()
            perm_idx = rng.permutation(len(X_perm))
            X_perm[:, idx] = X_perm[perm_idx, idx]
            pred, _ = predict_bundle(bundle, X_perm)
            deltas.append(mean_absolute_error(y, pred) - baseline_mae)
        rows.append({
            "feature": feat,
            "delta_mae": float(np.mean(deltas)) if deltas else np.nan,
            "sem": float(np.std(deltas, ddof=1) / np.sqrt(len(deltas))) if len(deltas) > 1 else 0.0,
        })
    if not rows:
        return pd.DataFrame(columns=["feature", "delta_mae", "sem"])
    return pd.DataFrame(rows).sort_values("delta_mae", ascending=False)


def compute_category_permutation_importance(
    bundle,
    X,
    y,
    feature_names,
    category_map,
    n_repeats=10,
    seed=RANDOM_STATE,
    min_features=5,
):
    if bundle is None or not category_map:
        return pd.DataFrame(columns=["category", "delta_mae", "sem", "n_features"])
    feature_to_idx = {name: i for i, name in enumerate(feature_names)}
    X_base = np.asarray(X, dtype=float)
    baseline_pred, _ = predict_bundle(bundle, X_base)
    baseline_mae = mean_absolute_error(y, baseline_pred)
    rng = np.random.default_rng(seed)
    rows = []
    for cat, feats in category_map.items():
        idxs = [feature_to_idx[f] for f in feats if f in feature_to_idx]
        if len(idxs) < min_features:
            continue
        deltas = []
        for _ in range(n_repeats):
            X_perm = X_base.copy()
            perm_idx = rng.permutation(len(X_perm))
            X_perm[:, idxs] = X_perm[perm_idx][:, idxs]
            pred, _ = predict_bundle(bundle, X_perm)
            deltas.append(mean_absolute_error(y, pred) - baseline_mae)
        rows.append({
            "category": cat,
            "delta_mae": float(np.mean(deltas)) if deltas else np.nan,
            "sem": float(np.std(deltas, ddof=1) / np.sqrt(len(deltas))) if len(deltas) > 1 else 0.0,
            "n_features": len(idxs),
        })
    if not rows:
        return pd.DataFrame(columns=["category", "delta_mae", "sem", "n_features"])
    return pd.DataFrame(rows).sort_values("delta_mae", ascending=False)


N_RESAMPLES = 30
selection_freq = compute_selection_frequency(
    X_all, y_all, feature_names,
    n_splits=N_RESAMPLES,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)
selection_freq = selection_freq.sort_values(ascending=False)

if bundle_full is None:
    bundle_full = fit_bundle(X_all, y_all, feature_names, use_gbr=True)


vt_mask = bundle_full["vt"].get_support()
coef = pd.Series(0.0, index=feature_names)
coef.loc[np.array(feature_names)[vt_mask]] = bundle_full["enet"].coef_
nonzero = coef[coef != 0]

primary_categories = pd.Series([categorize_primary(name) for name in feature_names], index=feature_names)
category_counts = primary_categories.value_counts().sort_values(ascending=False)
selected_category_counts = primary_categories.reindex(nonzero.index).value_counts().sort_values(ascending=False)
if selected_category_counts.empty:
    domain_summary = category_counts
else:
    domain_summary = selected_category_counts

corrs = {}
for col in feature_names:
    x = X_model_rescaled[col]
    valid = x.notna() & pd.notna(y_model)
    if valid.sum() > 2:
        x_valid = x[valid]
        if x_valid.std() > 0:
            corrs[col] = np.corrcoef(x_valid, y_model[valid])[0, 1]
corrs = pd.Series(corrs).dropna()

corrs_df = pd.DataFrame({
    "corr": corrs,
    "category": primary_categories.reindex(corrs.index),
}).dropna()

corr_order = (
    corrs_df.groupby("category")["corr"].apply(lambda s: s.abs().median()).sort_values().index
    if not corrs_df.empty
    else []
)

bands = ["delta", "theta", "alpha", "beta", "gamma"]
band_rows = []
for band in bands:
    band_idx = [f for f in corrs.index if band in f.lower()]
    if band_idx:
        vals = corrs.loc[band_idx]
        band_rows.append({
            "band": band.upper(),
            "mean_corr": vals.mean(),
            "sem_corr": vals.sem(),
            "n": len(vals),
        })
band_summary = pd.DataFrame(band_rows)

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(imputer.fit_transform(X_model_rescaled))
pca = PCA(n_components=2, random_state=RANDOM_STATE)
pca_scores = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame({
    "pc1": pca_scores[:, 0],
    "pc2": pca_scores[:, 1],
    "age": y_model,
})

stability_df = pd.DataFrame({
    "feature": selection_freq.index,
    "selection_freq": selection_freq.values,
})
stability_df["abs_coef"] = coef.abs().reindex(stability_df["feature"]).fillna(0).to_numpy()
stability_df["category"] = primary_categories.reindex(stability_df["feature"]).fillna("Unknown").to_numpy()
stability_df = stability_df.sort_values("selection_freq", ascending=False)

N_PERM_REPEATS = 10
N_PERM_FEATURES = 25
N_PERM_CATEGORY_MIN = 10

top_by_coef = coef.abs().sort_values(ascending=False).head(N_PERM_FEATURES).index.tolist()
top_by_stability = stability_df["feature"].head(N_PERM_FEATURES).tolist()
perm_features = []
for feat in top_by_coef + top_by_stability:
    if feat in perm_features:
        continue
    perm_features.append(feat)
    if len(perm_features) >= N_PERM_FEATURES:
        break

if bundle_eval is None:
    perm_feature_df = pd.DataFrame(columns=["feature", "delta_mae", "sem"])
    perm_category_df = pd.DataFrame(columns=["category", "delta_mae", "sem", "n_features"])
else:
    perm_feature_df = compute_permutation_importance(
        bundle_eval,
        X_test,
        y_test,
        feature_names,
        perm_features,
        n_repeats=N_PERM_REPEATS,
    )
    category_map = {}
    for cat in primary_categories.unique():
        feats = primary_categories[primary_categories == cat].index.tolist()
        category_map[cat] = feats
    perm_category_df = compute_category_permutation_importance(
        bundle_eval,
        X_test,
        y_test,
        feature_names,
        category_map,
        n_repeats=N_PERM_REPEATS,
        min_features=N_PERM_CATEGORY_MIN,
    )


# %%
def plot_age_distribution(ax, label=None):
    y_vals = pd.Series(y_model).dropna()
    if y_vals.empty:
        ax.text(0.5, 0.5, "No age data", ha="center", va="center")
    else:
        sns.histplot(y_vals, bins=20, kde=False, ax=ax, color=PALETTE["blue"], stat="percent")
        ax.axvline(y_vals.mean(), color=PALETTE["orange"], linestyle="--", linewidth=1)
        ax.set_xlabel("Age")
        ax.set_ylabel("Percent")
    ax.set_title("Age Distribution")
    style_axis(ax)
    if label:
        panel_label(ax, label)


def plot_sex_distribution(ax, label=None):
    sex_cols = [c for c in df_model.columns if c.lower().startswith("sex_")]
    if not sex_cols:
        ax.text(0.5, 0.5, "No sex columns", ha="center", va="center")
    else:
        sex_vals = df_model[sex_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        sex_counts = (sex_vals > 0.5).sum().sort_values(ascending=True)
        labels = [c.replace("sex_", "").replace("_", " ").title() for c in sex_counts.index]
        ax.barh(labels, sex_counts.values, color=PALETTE["teal"])
        ax.set_xlabel("Count")
    ax.set_title("Sex Distribution")
    style_axis(ax)
    if label:
        panel_label(ax, label)


def plot_feature_category_composition(ax, label=None):
    if domain_summary.empty:
        ax.text(0.5, 0.5, "No feature categories", ha="center", va="center")
    else:
        total = domain_summary.sum()
        left = 0.0
        for cat, val in domain_summary.items():
            width = 100 * val / total if total else 0
            ax.barh([0], width, left=left, color=CATEGORY_PALETTE.get(cat, PALETTE["gray"]), height=0.5)
            left += width
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("Percent of selected features")
        labels = [f"{cat} {100 * val / total:.0f}%" for cat, val in domain_summary.items()]
        ax.legend(labels, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.set_title("Feature Category Composition")
    style_axis(ax)
    if label:
        panel_label(ax, label)


def plot_predicted_vs_chronological_age(ax, label=None):
    if len(y_test) == 0:
        ax.text(0.5, 0.5, "No test data", ha="center", va="center")
    else:
        xx = np.linspace(np.min(y_test), np.max(y_test), 100)
        ax.scatter(y_test, pred_post, s=24, color=PALETTE["teal"], alpha=0.6, edgecolor="white", linewidth=0.3)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle="--", color=PALETTE["light_gray"], linewidth=1.2)
        slope, intercept, _, _, _ = stats.linregress(y_test, pred_post)
        ax.plot(xx, intercept + slope * xx, color=PALETTE["orange"], linewidth=2.2)
        ax.set_xlabel("Chronological age")
        ax.set_ylabel("Predicted age")
        ax.set_aspect("equal", adjustable="box")
    ax.set_title("Predicted vs Chronological Age")
    style_axis(ax)
    if label:
        panel_label(ax, label)


def plot_pca_structure_age(ax, label=None):
    if "pca_df" not in globals() or pca_df.empty:
        ax.text(0.5, 0.5, "No PCA data", ha="center", va="center")
    else:
        sc = ax.scatter(pca_df["pc1"], pca_df["pc2"], c=pca_df["age"], cmap=AGE_CMAP, s=14, alpha=0.7)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="Age")
    ax.set_title("PCA Structure (Age)")
    style_axis(ax)
    if label:
        panel_label(ax, label)


def plot_calibration_curve(ax, label=None):
    calib_df = pd.DataFrame({"true": y_test, "pred": pred_post}).dropna()
    if len(calib_df) < 5:
        ax.text(0.5, 0.5, "Not enough calibration data", ha="center", va="center")
    else:
        calib_df["bin"] = pd.qcut(calib_df["pred"], q=8, duplicates="drop")
        calib_summary = calib_df.groupby("bin", observed=True).agg(
            pred_mean=("pred", "mean"),
            true_mean=("true", "mean"),
            true_sem=("true", lambda s: float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0),
            n=("true", "size"),
        )
        min_age = min(calib_summary["pred_mean"].min(), calib_summary["true_mean"].min())
        max_age = max(calib_summary["pred_mean"].max(), calib_summary["true_mean"].max())
        ax.plot([min_age, max_age], [min_age, max_age], linestyle="--", color=PALETTE["light_gray"])
        sizes = 60 + 140 * (calib_summary["n"] / calib_summary["n"].max())
        ax.errorbar(
            calib_summary["pred_mean"],
            calib_summary["true_mean"],
            yerr=calib_summary["true_sem"],
            fmt="none",
            ecolor=PALETTE["slate"],
            capsize=3,
            linewidth=1,
        )
        ax.scatter(calib_summary["pred_mean"], calib_summary["true_mean"], s=sizes, color=PALETTE["teal"], edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Mean predicted age")
        ax.set_ylabel("Mean true age")
    ax.set_title("Calibration Curve")
    style_axis(ax)
    if label:
        panel_label(ax, label)


def plot_mae_by_age_bin(ax, label=None):
    if mae_by_bin.empty:
        ax.text(0.5, 0.5, "No MAE by age bin", ha="center", va="center")
    else:
        ax.bar(mae_by_bin["label"], mae_by_bin["mae"], color=PALETTE["blue"], alpha=0.85)
        ax.errorbar(
            mae_by_bin["label"],
            mae_by_bin["mae"],
            yerr=mae_by_bin["sem"],
            fmt="none",
            ecolor=PALETTE["slate"],
            capsize=3,
            linewidth=1,
        )
        ax.set_xlabel("Age bin")
        ax.set_ylabel("MAE (years)")
        ax.tick_params(axis="x", rotation=45)
    ax.set_title("MAE by Age Bin")
    style_axis(ax)
    if label:
        panel_label(ax, label)


def plot_residual_distribution(ax, label=None):
    if residuals_df.empty:
        ax.text(0.5, 0.5, "No residual data", ha="center", va="center")
    else:
        sns.violinplot(
            data=residuals_df,
            x="age_bin",
            y="residual",
            ax=ax,
            color=PALETTE["teal"],
            inner="quartile",
            cut=0,
        )
        sns.stripplot(
            data=residuals_df,
            x="age_bin",
            y="residual",
            ax=ax,
            color=PALETTE["ink"],
            size=2,
            jitter=0.2,
            alpha=0.4,
        )
        ax.axhline(0, color=PALETTE["slate"], linestyle="--", linewidth=1)
        ax.set_xlabel("Age bin")
        ax.set_ylabel("Residual (years)")
        ax.tick_params(axis="x", rotation=45)
    ax.set_title("Residual Distribution")
    style_axis(ax)
    if label:
        panel_label(ax, label)


def plot_figure3_feature_selection_by_category(ax, label=None):
    counts = selected_category_counts if not selected_category_counts.empty else category_counts
    if counts.empty:
        ax.text(0.5, 0.5, "No feature categories", ha="center", va="center")
    else:
        counts = counts.sort_values()
        colors = [CATEGORY_PALETTE.get(cat, PALETTE["gray"]) for cat in counts.index]
        ax.barh(counts.index, counts.values, color=colors)
        ax.set_xlabel("Count")
    ax.set_title("Feature selection by category")
    style_axis(ax)
    if label:
        panel_label(ax, label)


def plot_figure3_category_permutation_importance(ax, label=None):
    if perm_category_df.empty:
        ax.text(0.5, 0.5, "No category importance data", ha="center", va="center")
    else:
        df_plot = perm_category_df.sort_values("delta_mae", ascending=True)
        ax.barh(
            df_plot["category"],
            df_plot["delta_mae"],
            xerr=df_plot["sem"],
            color=PALETTE["teal"],
            alpha=0.85,
        )
        for i, row in enumerate(df_plot.itertuples(index=False)):
            ax.text(
                row.delta_mae + (df_plot["delta_mae"].max() * 0.02 if len(df_plot) else 0.02),
                i,
                f"n={row.n_features}",
                va="center",
                fontsize=8,
            )
        ax.set_xlabel("Delta MAE (years)")
    ax.set_title("Category permutation importance")
    style_axis(ax)
    if label:
        panel_label(ax, label)


def plot_figure3_eeg_band_aging_signatures(ax, label=None):
    if band_summary.empty:
        ax.text(0.5, 0.5, "No EEG band features", ha="center", va="center")
    else:
        x = np.arange(len(band_summary))
        colors = [PALETTE["teal"] if v >= 0 else PALETTE["red"] for v in band_summary["mean_corr"]]
        ax.bar(x, band_summary["mean_corr"], color=colors, alpha=0.85)
        ax.errorbar(
            x,
            band_summary["mean_corr"],
            yerr=band_summary["sem_corr"].fillna(0.0),
            fmt="none",
            ecolor=PALETTE["slate"],
            capsize=3,
            linewidth=1,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(band_summary["band"])
        ax.axhline(0, color=PALETTE["slate"], linestyle="--", linewidth=1)
        ax.set_ylabel("Mean corr(age)")
    ax.set_title("EEG band aging signatures")
    style_axis(ax)
    if label:
        panel_label(ax, label)


def plot_figure3_coefficient_direction_by_category(ax, label=None):
    coef_sign_df = pd.DataFrame({
        "coef": nonzero,
        "category": [categorize_primary(name) for name in nonzero.index],
    })
    if coef_sign_df.empty:
        ax.text(0.5, 0.5, "No selected features", ha="center", va="center")
    else:
        coef_sign_df["sign"] = np.where(coef_sign_df["coef"] >= 0, "Positive", "Negative")
        sign_counts = coef_sign_df.groupby(["category", "sign"], observed=True).size().unstack(fill_value=0)
        sign_counts["total"] = sign_counts.sum(axis=1)
        sign_counts = sign_counts[sign_counts["total"] > 0]
        sign_counts = sign_counts.sort_values("total", ascending=True)
        pos_pct = 100 * sign_counts.get("Positive", 0) / sign_counts["total"]
        neg_pct = 100 * sign_counts.get("Negative", 0) / sign_counts["total"]
        ax.barh(sign_counts.index, -neg_pct, color=PALETTE["red"], alpha=0.8, label="Negative")
        ax.barh(sign_counts.index, pos_pct, color=PALETTE["green"], alpha=0.8, label="Positive")
        ax.axvline(0, color=PALETTE["slate"], linewidth=1)
        ax.set_xlim(-100, 100)
        ax.set_xlabel("Coefficient sign share (%)")
        ax.legend(frameon=False, loc="upper right")
    ax.set_title("Coefficient direction by category")
    style_axis(ax)
    if label:
        panel_label(ax, label)


def plot_figure3_importance_vs_stability(ax, label=None):
    if perm_category_df.empty or stability_df.empty:
        ax.text(0.5, 0.5, "No importance or stability data", ha="center", va="center")
    else:
        perm_summary = perm_category_df.set_index("category")["delta_mae"]
        stab_summary = stability_df.groupby("category", observed=True)["selection_freq"].mean()
        counts = selected_category_counts if not selected_category_counts.empty else category_counts
        bubble_df = pd.DataFrame({
            "delta_mae": perm_summary,
            "stability": stab_summary,
            "count": counts,
        }).dropna()
        if bubble_df.empty:
            ax.text(0.5, 0.5, "No overlapping categories", ha="center", va="center")
        else:
            sizes = 80 + 220 * (bubble_df["count"] / bubble_df["count"].max())
            colors = [CATEGORY_PALETTE.get(cat, PALETTE["gray"]) for cat in bubble_df.index]
            ax.scatter(
                bubble_df["stability"],
                bubble_df["delta_mae"],
                s=sizes,
                color=colors,
                alpha=0.8,
                edgecolor="white",
                linewidth=0.6,
            )
            for cat, row in bubble_df.iterrows():
                ax.text(row["stability"] + 0.01, row["delta_mae"], cat, fontsize=8, va="center")
            ax.set_xlabel("Mean selection frequency")
            ax.set_ylabel("Permutation importance (delta MAE)")
    ax.set_title("Importance vs stability")
    style_axis(ax)
    if label:
        panel_label(ax, label)


def plot_figure3_age_corr_vs_coefficient_strength(ax, label=None):
    if corrs_df.empty or nonzero.empty:
        ax.text(0.5, 0.5, "No correlation or coefficient data", ha="center", va="center")
    else:
        corr_summary = corrs_df.groupby("category", observed=True)["corr"].mean()
        coef_summary = nonzero.abs().groupby([categorize_primary(name) for name in nonzero.index]).mean()
        counts = selected_category_counts if not selected_category_counts.empty else category_counts
        align_df = pd.DataFrame({
            "corr": corr_summary,
            "coef": coef_summary,
            "count": counts,
        }).dropna()
        if align_df.empty:
            ax.text(0.5, 0.5, "No overlapping categories", ha="center", va="center")
        else:
            sizes = 80 + 220 * (align_df["count"] / align_df["count"].max())
            colors = [CATEGORY_PALETTE.get(cat, PALETTE["gray"]) for cat in align_df.index]
            ax.scatter(
                align_df["corr"],
                align_df["coef"],
                s=sizes,
                color=colors,
                alpha=0.8,
                edgecolor="white",
                linewidth=0.6,
            )
            ax.axvline(0, color=PALETTE["slate"], linestyle="--", linewidth=1)
            for cat, row in align_df.iterrows():
                ax.text(row["corr"] + 0.01, row["coef"], cat, fontsize=8, va="center")
            ax.set_xlabel("Mean corr(age)")
            ax.set_ylabel("Mean |coef|")
    ax.set_title("Age correct vs coefficient strength")
    style_axis(ax)
    if label:
        panel_label(ax, label)


def plot_figure3_category_signal_across_methods(ax, label=None):
    def _category_share(feature_list):
        cats = primary_categories.reindex(feature_list).fillna("Unknown")
        counts = cats.value_counts()
        total = counts.sum()
        if total == 0:
            return pd.Series(dtype=float)
        return counts / total

    method_shares = {}
    top_n = 20
    if not coef.empty:
        top_coef = coef.abs().sort_values(ascending=False).head(top_n).index
        method_shares["Top |coef|"] = _category_share(top_coef)
    if not perm_feature_df.empty:
        top_perm = perm_feature_df["feature"].head(top_n)
        method_shares["Top perm"] = _category_share(top_perm)
    if not stability_df.empty:
        top_stab = stability_df.sort_values("selection_freq", ascending=False).head(top_n)["feature"]
        method_shares["Top stability"] = _category_share(top_stab)

    if not method_shares:
        ax.text(0.5, 0.5, "No method overlap data", ha="center", va="center")
    else:
        share_df = pd.DataFrame(method_shares).fillna(0.0) * 100
        share_df = share_df.loc[share_df.sum(axis=1).sort_values().index]
        sns.heatmap(
            share_df,
            ax=ax,
            cmap="YlGnBu",
            annot=True,
            fmt=".0f",
            cbar_kws={"label": "Share of features (%)"},
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
    ax.set_title("Category signal across methods")
    style_axis(ax)
    if label:
        panel_label(ax, label)


def plot_figure3_selected_strength_by_category(ax, label=None):
    ax_b_data = pd.DataFrame({
        "abs_coef": nonzero.abs(),
        "category": [categorize_primary(name) for name in nonzero.index],
    })
    if ax_b_data.empty:
        ax.text(0.5, 0.5, "No selected features", ha="center", va="center")
    else:
        order = [cat for cat in category_counts.index if cat in ax_b_data["category"].unique()]
        sns.boxplot(
            data=ax_b_data,
            x="abs_coef",
            y="category",
            order=order,
            hue="category",
            ax=ax,
            palette=CATEGORY_PALETTE,
            showfliers=False,
            dodge=False,
            legend=False,
        )
        ax.set_xlabel("|coefficient|")
        ax.set_ylabel("")
    ax.set_title("Selected Feature Strength by Category")
    style_axis(ax)
    if label:
        panel_label(ax, label)


def plot_figure3_age_corr_by_category(ax, label=None):
    if corrs_df.empty:
        ax.text(0.5, 0.5, "No correlation data", ha="center", va="center")
    else:
        sns.violinplot(
            data=corrs_df,
            y="category",
            x="corr",
            order=list(corr_order),
            hue="category",
            ax=ax,
            palette=CATEGORY_PALETTE,
            cut=0,
            dodge=False,
            legend=False,
        )
        ax.axvline(0, color=PALETTE["slate"], linestyle="--", linewidth=1)
        ax.set_xlabel("Correlation with age")
        ax.set_ylabel("")
    ax.set_title("Age Correlation by Category")
    style_axis(ax)
    if label:
        panel_label(ax, label)


def plot_race_distribution(ax, label=None):
    race_cols = [c for c in df_model.columns if c.lower().startswith("race_")]
    if not race_cols:
        ax.text(0.5, 0.5, "No race columns", ha="center", va="center")
    else:
        race_vals = df_model[race_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        race_counts = (race_vals > 0.5).sum().sort_values(ascending=True)
        labels = [c.replace("race_", "").replace("_", " ").title() for c in race_counts.index]
        ax.barh(labels, race_counts.values, color=PALETTE["purple"])
        ax.set_xlabel("Count")
    ax.set_title("Race Distribution")
    style_axis(ax)
    if label:
        panel_label(ax, label)


def plot_education_distribution(ax, label=None):
    edu_cols = [c for c in df_model.columns if c.lower().startswith("education_")]
    if not edu_cols:
        ax.text(0.5, 0.5, "No education columns", ha="center", va="center")
    else:
        edu_vals = df_model[edu_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        edu_counts = (edu_vals > 0.5).sum().sort_values(ascending=True)
        labels = [c.replace("education_", "").replace("_", " ").title() for c in edu_counts.index]
        ax.barh(labels, edu_counts.values, color=PALETTE["gold"])
        ax.set_xlabel("Count")
    ax.set_title("Education Distribution")
    style_axis(ax)
    if label:
        panel_label(ax, label)


# %%
output_dir = BASE / "figures"
output_dir.mkdir(exist_ok=True)


def save_panel(name, label, plot_func):
    fig, ax = _new_panel()
    plot_func(ax, label=label)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.svg", bbox_inches="tight")
    display(fig)
    plt.close(fig)


panels = [
    ("figure1_panel_f", "1f", plot_age_distribution),
    ("figure1_panel_g", "1g", plot_sex_distribution),
    ("figure1_panel_h", "1h", plot_feature_category_composition),
    ("figure2_panel_a", "2a", plot_predicted_vs_chronological_age),
    ("figure2_panel_b", "2b", plot_pca_structure_age),
    ("figure2_panel_c", "2c", plot_calibration_curve),
    ("figure2_panel_d", "2d", plot_mae_by_age_bin),
    ("figure2_panel_e", "2e", plot_residual_distribution),
    ("figure3_panel_a", "3a", plot_figure3_category_permutation_importance),
    ("figure3_panel_b", "3b", plot_figure3_coefficient_direction_by_category),
    ("figure3_panel_c", "3c", plot_figure3_eeg_band_aging_signatures),
    ("figure3_panel_d", "3d", plot_figure3_importance_vs_stability),
    ("figure3_panel_e", "3e", plot_figure3_category_signal_across_methods),
    ("figure3_panel_f", "3f", plot_figure3_age_corr_by_category),
    ("figure3_panel_g", "3g", plot_figure3_age_corr_vs_coefficient_strength),
    ("figure3_panel_h", "3h", plot_figure3_feature_selection_by_category),
    ("supplemental_panel_s1", "S1", plot_race_distribution),
    ("supplemental_panel_s2", "S2", plot_education_distribution),
    ("supplemental_panel_s4", "S4", plot_figure3_selected_strength_by_category),
]

for name, label, plot_func in panels:
    save_panel(name, label, plot_func)

plt.show()
