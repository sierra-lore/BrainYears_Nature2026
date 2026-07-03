#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PowerTransformer, StandardScaler, PolynomialFeatures
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "model_simulated_data.csv"
OUT_BUNDLE = BASE / "model_bundle.joblib"

RANDOM_STATE = 52
STRATIFY_BINS = 18
CLIP_PERCENTILE = 0.5
VARIANCE_THRESHOLD = 1e-7

ELASTICNET_ALPHA = 0.325
ELASTICNET_L1_RATIO = 0.235
ELASTICNET_SELECTION = "random"

GBR_N_ESTIMATORS = 185
GBR_LEARNING_RATE = 0.0175
GBR_MAX_DEPTH = 2
GBR_SUBSAMPLE = 0.5
GBR_MIN_SAMPLES_LEAF = 60
GBR_LOSS = "huber"

BIAS_POLY_DEGREE = 3


def fit_winsor_clip_bounds(train_features, clip_percentile):
    lo = np.percentile(train_features, clip_percentile, axis=0)
    hi = np.percentile(train_features, 100 - clip_percentile, axis=0)
    return lo, hi


def apply_winsor_clipping(features, lo, hi):
    return np.clip(features, lo, hi)


def fit_two_stage_model_v2(X_train_raw_clipped, X_train_scaled, y_train):
    """
    - ElasticNet trains on scaled features (after vt->clip->pt->ss)
    - GBR trains on [RAW_CLIPPED features, ElasticNet prediction] to predict residuals
    """
    enet = ElasticNet(
        alpha=ELASTICNET_ALPHA,
        l1_ratio=ELASTICNET_L1_RATIO,
        selection=ELASTICNET_SELECTION,
        random_state=RANDOM_STATE,
        max_iter=100000,
        tol=1e-5,
    ).fit(X_train_scaled, y_train)

    enet_pred = enet.predict(X_train_scaled)
    residuals = y_train - enet_pred

    X_gbr = np.column_stack([X_train_raw_clipped, enet_pred])

    gbr = GradientBoostingRegressor(
        n_estimators=GBR_N_ESTIMATORS,
        learning_rate=GBR_LEARNING_RATE,
        max_depth=GBR_MAX_DEPTH,
        subsample=GBR_SUBSAMPLE,
        min_samples_leaf=GBR_MIN_SAMPLES_LEAF,
        loss=GBR_LOSS,
        random_state=RANDOM_STATE,
    ).fit(X_gbr, residuals)

    return enet, gbr


def predict_two_stage_model_v2(enet, gbr, X_raw_clipped, X_scaled):
    enet_pred = enet.predict(X_scaled)
    X_gbr = np.column_stack([X_raw_clipped, enet_pred])
    corr = gbr.predict(X_gbr)
    return enet_pred + corr


# load data
df = pd.read_csv(DATA_PATH)

# keep only ages 18–89.99
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df = df[(df["age"] >= 18.0) & (df["age"] <= 89.99)].copy()
print(f"Loaded {len(df)} samples after age filtering [18.0, 89.99]")

# Separate features/target
X_all_cols = df.drop(columns=["age"])
y_all = df["age"].to_numpy(float)

# Drop rows with ANY NaN in feature columns
not_nan_mask = ~X_all_cols.isna().any(axis=1).to_numpy()
X_filtered = X_all_cols.loc[not_nan_mask].reset_index(drop=True)
y_all = y_all[not_nan_mask]

# Numeric features only (and keep the column order!)
Xdf = X_filtered.select_dtypes(include=[np.number]).copy()
feats = Xdf.columns.to_list()
X_all = Xdf.to_numpy(float)

print(f"After NaN filtering: {len(X_all)} samples, {X_all.shape[1]} numeric features")

def make_stratify_bins(y, n_bins):
    if not n_bins or n_bins < 2 or len(y) < n_bins * 2:
        return None
    try:
        bins = pd.qcut(pd.Series(y), q=n_bins, duplicates="drop")
        if (bins.value_counts() >= 2).all():
            return bins
    except Exception as exc:
        print(f"   Stratify binning skipped: {type(exc).__name__}: {exc}")
    return None


print(f"\n📊 Creating up to {STRATIFY_BINS} age bins for stratified splitting...")
stratify_param = make_stratify_bins(y_all, STRATIFY_BINS)
if stratify_param is None:
    print("   Using an unstratified split for the available sample size.")

print(f"\n🔀 Splitting data (test_size=0.20, random_state={RANDOM_STATE})...")
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.20, random_state=RANDOM_STATE, stratify=stratify_param
)

# preprocess (fit on train only)
vt = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
X_train_vt = vt.fit_transform(X_train_raw)
X_test_vt = vt.transform(X_test_raw)
print(f"After variance filter: {X_train_vt.shape[1]} features")

clip_lo, clip_hi = fit_winsor_clip_bounds(X_train_vt, CLIP_PERCENTILE)
X_train_clipped = apply_winsor_clipping(X_train_vt, clip_lo, clip_hi)
X_test_clipped = apply_winsor_clipping(X_test_vt, clip_lo, clip_hi)

pt = PowerTransformer(method="yeo-johnson", standardize=True)
X_train_pt = pt.fit_transform(X_train_clipped)
X_test_pt = pt.transform(X_test_clipped)

ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train_pt)
X_test_scaled = ss.transform(X_test_pt)

# fit models
enet, gbr = fit_two_stage_model_v2(X_train_clipped, X_train_scaled, y_train)

# evaluate before bias correction
train_pred_raw = predict_two_stage_model_v2(enet, gbr, X_train_clipped, X_train_scaled)
test_pred_raw = predict_two_stage_model_v2(enet, gbr, X_test_clipped, X_test_scaled)

print("\n" + "="*60)
print("ELASTICNET+GBR (before bias correction)")
print("="*60)
print(f"Train MAE: {mean_absolute_error(y_train, train_pred_raw):.4f}")
print(f"Train r: {pearsonr(train_pred_raw, y_train)[0]:.4f}")
print(f"Test MAE: {mean_absolute_error(y_test, test_pred_raw):.4f}")
print(f"Test r: {pearsonr(test_pred_raw, y_test)[0]:.4f}")

# bias correction: fit error ~ poly(predicted_age)
oof_error = train_pred_raw - y_train
poly = PolynomialFeatures(degree=BIAS_POLY_DEGREE, include_bias=False)
A_train = poly.fit_transform(train_pred_raw.reshape(-1, 1))
bias_model = LinearRegression().fit(A_train, oof_error)

# corrected eval
train_bias = bias_model.predict(A_train)
train_pred_corr = train_pred_raw - train_bias

A_test = poly.transform(test_pred_raw.reshape(-1, 1))
test_bias = bias_model.predict(A_test)
test_pred_corr = test_pred_raw - test_bias

print("\n" + "="*60)
print("FINAL (after bias correction)")
print("="*60)
print(f"Train MAE: {mean_absolute_error(y_train, train_pred_corr):.4f}")
print(f"Train r: {pearsonr(train_pred_corr, y_train)[0]:.4f}")
print(f"Test MAE: {mean_absolute_error(y_test, test_pred_corr):.4f}")
print(f"Test r: {pearsonr(test_pred_corr, y_test)[0]:.4f}")
print("="*60)

# SAVE BUNDLE (as a dict)
bundle = dict(
    feats=feats,
    vt=vt,
    clip_lo=clip_lo,
    clip_hi=clip_hi,
    pt=pt,
    ss=ss,
    enet=enet,
    gbr=gbr,
    poly=poly,
    bias=bias_model,
    meta=dict(
        data_path=str(DATA_PATH),
        random_state=RANDOM_STATE,
        stratify_bins=STRATIFY_BINS,
        clip_percentile=CLIP_PERCENTILE,
        variance_threshold=VARIANCE_THRESHOLD,
        enet_alpha=ELASTICNET_ALPHA,
        enet_l1_ratio=ELASTICNET_L1_RATIO,
        gbr_params=dict(
            n_estimators=GBR_N_ESTIMATORS,
            learning_rate=GBR_LEARNING_RATE,
            max_depth=GBR_MAX_DEPTH,
            subsample=GBR_SUBSAMPLE,
            min_samples_leaf=GBR_MIN_SAMPLES_LEAF,
            loss=GBR_LOSS,
        ),
        bias_poly_degree=BIAS_POLY_DEGREE,
    )
)

joblib.dump(bundle, OUT_BUNDLE)
print(f"\n✅ Saved model bundle to: {OUT_BUNDLE}")
