# app.py — Stock Price Prediction Dashboard
# Run: streamlit run app.py
import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Stock Price Prediction", layout="wide")

CUSTOM_CSS = '''
<style>
section[data-testid="stSidebar"] > div {
  background: #111827;
  padding: 20px;
  color: #ffffff;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2 {
  color: #ffffff !important;
  font-weight: 700 !important;
}
.step-pill {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  padding: .55rem .8rem;
  margin: .3rem 0;
  border-radius: 10px;
  font-size: 15px;
  font-weight: 500;
  background: #1f2937;
  color: #e5e7eb;
  border: 1px solid #374151;
  cursor: pointer;
  transition: all 0.15s ease-in-out;
}
.step-pill .idx {
  width: 1.4rem;
  height: 1.4rem;
  border-radius: 999px;
  background: #4b5563;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: .75rem;
}
.step-pill.active {
  background: #2563eb;
  border-color: #3b82f6;
  color: #ffffff;
  box-shadow: 0 0 10px rgba(37, 99, 235, 0.4);
}
.step-pill.active .idx {
  background: white;
  color: #2563eb;
}
.step-pill:hover {
  transform: translateX(3px);
  background: #334155;
}
</style>
'''

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DATE_CANDIDATES = ["Date", "date", "Datetime", "datetime", "timestamp", "Timestamp"]
TARGET_CANDIDATES = ["Close", "Adj Close", "close", "adj_close"]

for key, default in {
    "df": None,
    "features_df": None,
    "prepared_df": None,
    "feature_target": None,
    "last_predictions": None,
    "future_forecast": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def readable_int(n):
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)


@st.cache_data(show_spinner=False)
def load_any(file_bytes: bytes, name: str) -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)
    if name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(bio)
    else:
        df = pd.read_csv(bio, low_memory=False)

    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if any(k in c.lower() for k in ["date", "time", "timestamp"]):
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
            except Exception:
                pass
    return df


@st.cache_data(show_spinner=False)
def compute_missing(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum().sort_values(ascending=False)
    return miss[miss > 0].to_frame("MissingCount")


def detect_first_present(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


@st.cache_data(show_spinner=False)
def prepare_dataset(df: pd.DataFrame, date_col: str | None = None) -> pd.DataFrame:
    out = df.copy()
    if date_col and date_col in out.columns:
        out = out.sort_values(date_col).reset_index(drop=True)

    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            continue
        if out[c].dtype == object:
            stripped = out[c].astype(str).str.replace(",", "", regex=False).str.strip()
            numeric = pd.to_numeric(stripped, errors="coerce")
            if numeric.notna().sum() > 0 and numeric.notna().sum() >= max(3, int(0.6 * len(out))):
                out[c] = numeric
    return out


@st.cache_data(show_spinner=False)
def make_lag_features(
    prepared_df: pd.DataFrame,
    target_col: str,
    lags=(1, 2, 3, 5, 10),
    windows=(5, 10, 20),
    include_return=True,
):
    if target_col not in prepared_df.columns:
        raise ValueError(f"Target column '{target_col}' was not found.")

    target_series = pd.to_numeric(prepared_df[target_col], errors="coerce")
    if target_series.notna().sum() < 10:
        raise ValueError(
            f"The selected target '{target_col}' does not contain enough numeric values for feature engineering."
        )

    out = pd.DataFrame(index=prepared_df.index)
    out[target_col] = target_series

    valid_lags = sorted({int(x) for x in lags if int(x) > 0})
    valid_windows = sorted({int(x) for x in windows if int(x) > 1})

    for lag in valid_lags:
        out[f"{target_col}_lag{lag}"] = target_series.shift(lag)

    for window in valid_windows:
        shifted = target_series.shift(1)
        out[f"{target_col}_sma{window}"] = shifted.rolling(window).mean()
        out[f"{target_col}_ema{window}"] = shifted.ewm(span=window, adjust=False).mean()
        out[f"{target_col}_rstd{window}"] = shifted.rolling(window).std()
        out[f"{target_col}_mom{window}"] = shifted / shifted.shift(window - 1) - 1.0

    if include_return:
        out["return_1"] = target_series.pct_change(1)

    extra_numeric = prepared_df.select_dtypes(include=[np.number]).copy()
    extra_numeric = extra_numeric.drop(columns=[target_col], errors="ignore")
    out = pd.concat([out, extra_numeric], axis=1)

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna().reset_index(drop=True)

    if out.empty:
        raise ValueError(
            "Feature engineering created no usable rows. Try smaller lags/windows or check whether the target column has valid numeric data."
        )

    return out


def to_csv_download(df, fname, label="⬇️ Download CSV"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, file_name=fname, mime="text/csv")


steps = [
    ("Upload Data", "📤"),
    ("Preview & Columns", "👀"),
    ("Missing Values", "🧩"),
    ("Feature Engineering", "🧪"),
    ("EDA", "📊"),
    ("Train & Evaluate", "🎯"),
    ("Forecast", "📈"),
    ("Export", "💾"),
]
step_titles = [s[0] for s in steps]

with st.sidebar:
    st.title("Stock Dashboard")
    current = st.radio("Navigate", step_titles, index=0, label_visibility="collapsed")
    st.markdown("### Steps")
    for idx, (title, icon) in enumerate(steps, start=1):
        cls = "step-pill active" if title == current else "step-pill"
        st.markdown(
            f'<div class="{cls}"><span class="idx">{idx}</span> {icon} {title}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.caption("Use time-based split for realistic evaluation.")

st.title("📈 Stock Price Prediction – E2E Dashboard")
st.caption("Upload → Preview → Missing → Feature Engineering → EDA → Train → Forecast → Export")

if current == "Upload Data":
    st.header("1) Upload Data")
    up = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx", "xls"])
    if up is not None:
        df = load_any(up.getvalue(), up.name)
        st.session_state["df"] = df
        st.session_state["features_df"] = None
        st.session_state["prepared_df"] = None
        st.session_state["last_predictions"] = None
        st.session_state["future_forecast"] = None
        st.success(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(10))
        to_csv_download(df, "uploaded_copy.csv")
    else:
        st.info("Upload a file to start.")

if current == "Preview & Columns":
    st.header("2) Preview & Columns")
    df = st.session_state["df"]
    if df is None:
        st.warning("Please upload a file first.")
    else:
        cols = list(df.columns)
        date_col = detect_first_present(cols, DATE_CANDIDATES)
        c1, c2 = st.columns([2, 1])
        with c1:
            chosen_date_col = st.selectbox(
                "Date column for ordering (optional)", [None] + cols,
                index=(cols.index(date_col) + 1) if date_col in cols else 0,
            )
        with c2:
            if st.button("Prepare dataset"):
                prepared_df = prepare_dataset(df, chosen_date_col)
                st.session_state["prepared_df"] = prepared_df
                st.success("Dataset prepared successfully.")

        prepared_df = st.session_state.get("prepared_df")
        active_df = prepared_df if prepared_df is not None else df

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rows", readable_int(active_df.shape[0]))
        with c2:
            st.metric("Columns", readable_int(active_df.shape[1]))
        with c3:
            st.metric("Numeric Cols", readable_int(active_df.select_dtypes(include=[np.number]).shape[1]))

        st.subheader("Top 10 rows")
        st.dataframe(active_df.head(10))
        st.subheader("Columns & Dtypes")
        st.dataframe(pd.DataFrame({"column": active_df.columns, "dtype": active_df.dtypes.astype(str)}))

if current == "Missing Values":
    st.header("3) Missing Values")
    df = st.session_state.get("prepared_df") or st.session_state.get("df")
    if df is None:
        st.warning("Please upload a file first.")
    else:
        miss = compute_missing(df)
        if miss.empty:
            st.success("No missing values detected 🎉")
        else:
            st.info("Per-column missing values")
            st.dataframe(miss)

if current == "Feature Engineering":
    st.header("4) Feature Engineering (lags, rolling stats)")
    raw_df = st.session_state["df"]
    prepared_df = st.session_state.get("prepared_df")

    if raw_df is None:
        st.warning("Please upload a file first.")
    else:
        base_df = prepared_df if prepared_df is not None else prepare_dataset(raw_df)
        cols = list(base_df.columns)
        tgt_default = detect_first_present(cols, TARGET_CANDIDATES)
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(base_df[c])]
        default_target = tgt_default if tgt_default in cols else (numeric_cols[0] if numeric_cols else cols[0])

        tgt = st.selectbox(
            "Target (e.g., Close)",
            cols,
            index=cols.index(default_target) if default_target in cols else 0,
        )
        lags = st.multiselect("Lags", [1, 2, 3, 5, 10, 15, 20, 30], default=[1, 2, 3, 5, 10])
        windows = st.multiselect("Rolling Windows", [5, 10, 20, 30, 50, 100], default=[5, 10, 20])

        target_numeric = pd.to_numeric(base_df[tgt], errors="coerce")
        st.caption(f"Numeric rows available in '{tgt}': {int(target_numeric.notna().sum())} / {len(base_df)}")

        if st.button("🧪 Build Features"):
            try:
                if not lags:
                    raise ValueError("Please select at least one lag.")
                if not windows:
                    raise ValueError("Please select at least one rolling window.")

                fe = make_lag_features(base_df, tgt, lags=tuple(lags), windows=tuple(windows))
                st.session_state["features_df"] = fe
                st.session_state["feature_target"] = tgt
                st.success(f"Feature matrix ready: {fe.shape[0]} rows × {fe.shape[1]} columns")
                st.dataframe(fe.head(10))
                to_csv_download(fe, "features.csv")
            except Exception as e:
                st.session_state["features_df"] = None
                st.error(f"Feature engineering failed: {e}")

if current == "EDA":
    st.header("5) EDA")
    df = st.session_state.get("prepared_df") or st.session_state.get("df")
    if df is None:
        st.warning("Please upload a file first.")
    else:
        eda_df = prepare_dataset(df)
        cols = list(eda_df.columns)
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(eda_df[c])]
        date_col = detect_first_present(cols, DATE_CANDIDATES)
        price_col = detect_first_present(cols, TARGET_CANDIDATES) or (numeric_cols[0] if numeric_cols else None)

        if not numeric_cols:
            st.error("No numeric columns found for EDA plots. Please check your uploaded dataset.")
        else:
            st.caption("Use the options below to plot every selected numeric column against every other selected numeric column.")

            if len(eda_df) <= 100:
                max_rows = len(eda_df)
                st.caption(f"Using all {len(eda_df)} rows for EDA plots.")
            else:
                max_rows = st.slider(
                    "Rows to use for EDA plots",
                    100,
                    min(10000, len(eda_df)),
                    min(2000, len(eda_df)),
                    step=100,
                )
            plot_df = eda_df.tail(max_rows).copy()

            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Time Series",
                "🔁 Each vs Each",
                "📊 Distributions",
                "🧮 Correlation",
            ])

            with tab1:
                c1, c2 = st.columns(2)
                with c1:
                    tcol = st.selectbox("Time axis", [None] + cols, index=(cols.index(date_col) + 1) if date_col in cols else 0)
                with c2:
                    selected_series_cols = st.multiselect(
                        "Numeric columns to plot over time/index",
                        numeric_cols,
                        default=[price_col] if price_col in numeric_cols else numeric_cols[:1],
                    )

                for ycol in selected_series_cols:
                    y = pd.to_numeric(plot_df[ycol], errors="coerce")
                    if tcol:
                        valid = pd.DataFrame({tcol: plot_df[tcol], ycol: y}).dropna()
                        x_values = valid[tcol]
                        y_values = valid[ycol]
                        xlabel = tcol
                    else:
                        valid = y.dropna().reset_index(drop=True)
                        x_values = valid.index
                        y_values = valid.values
                        xlabel = "Index"

                    if len(valid) > 1:
                        fig, ax = plt.subplots(figsize=(9, 4))
                        ax.plot(x_values, y_values)
                        ax.set_xlabel(xlabel)
                        ax.set_ylabel(ycol)
                        ax.set_title(f"{ycol} over {xlabel}")
                        ax.grid(True, alpha=0.25)
                        st.pyplot(fig)
                        plt.close(fig)

            with tab2:
                st.subheader("Each numeric column with each numeric column")
                st.caption("Select a small group of numeric columns. The app will create scatter plots for every pair.")

                default_pair_cols = numeric_cols[: min(5, len(numeric_cols))]
                pair_cols = st.multiselect(
                    "Columns for pairwise EDA",
                    numeric_cols,
                    default=default_pair_cols,
                    key="eda_pair_cols",
                )

                pair_count = len(pair_cols) * (len(pair_cols) - 1) // 2
                st.info(f"Selected columns: {len(pair_cols)} | Pairwise plots to generate: {pair_count}")

                if len(pair_cols) < 2:
                    st.warning("Select at least two numeric columns for each-vs-each plots.")
                elif len(pair_cols) > 8:
                    st.warning("Please select 8 or fewer columns at a time to keep the dashboard fast and avoid memory errors.")
                else:
                    clean_pair_df = plot_df[pair_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
                    for i, xcol in enumerate(pair_cols):
                        for ycol in pair_cols[i + 1:]:
                            valid = clean_pair_df[[xcol, ycol]].dropna()
                            if valid.shape[0] < 2:
                                continue

                            fig, ax = plt.subplots(figsize=(6.5, 4.2))
                            ax.scatter(valid[xcol], valid[ycol], alpha=0.65, s=18)
                            ax.set_xlabel(xcol)
                            ax.set_ylabel(ycol)
                            ax.set_title(f"{xcol} vs {ycol}")
                            ax.grid(True, alpha=0.25)
                            st.pyplot(fig)
                            plt.close(fig)

            with tab3:
                st.subheader("Distribution plots")
                default_dist_cols = numeric_cols[: min(4, len(numeric_cols))]
                dist_cols = st.multiselect(
                    "Columns for histogram distribution",
                    numeric_cols,
                    default=default_dist_cols,
                    key="eda_dist_cols",
                )

                for col in dist_cols:
                    values = pd.to_numeric(plot_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                    if values.empty:
                        continue
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.hist(values, bins=30)
                    ax.set_xlabel(col)
                    ax.set_ylabel("Frequency")
                    ax.set_title(f"Distribution of {col}")
                    ax.grid(True, alpha=0.25)
                    st.pyplot(fig)
                    plt.close(fig)

            with tab4:
                st.subheader("Correlation heatmap")
                corr_cols = st.multiselect(
                    "Columns for correlation",
                    numeric_cols,
                    default=numeric_cols[: min(8, len(numeric_cols))],
                    key="eda_corr_cols",
                )

                if len(corr_cols) < 2:
                    st.warning("Select at least two numeric columns for correlation.")
                else:
                    corr = plot_df[corr_cols].apply(pd.to_numeric, errors="coerce").corr()
                    fig, ax = plt.subplots(figsize=(max(7, len(corr_cols) * 0.75), max(5, len(corr_cols) * 0.55)))
                    im = ax.imshow(corr, aspect="auto", vmin=-1, vmax=1)
                    ax.set_xticks(range(len(corr_cols)))
                    ax.set_yticks(range(len(corr_cols)))
                    ax.set_xticklabels(corr_cols, rotation=45, ha="right")
                    ax.set_yticklabels(corr_cols)
                    ax.set_title("Correlation Heatmap")
                    for row in range(len(corr_cols)):
                        for col in range(len(corr_cols)):
                            value = corr.iloc[row, col]
                            if pd.notna(value):
                                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

if current == "Train & Evaluate":
    st.header("6) Train & Evaluate")
    fe = st.session_state["features_df"]
    if fe is None or fe.empty:
        st.warning("Please build features first (previous step).")
    else:
        default_tgt = st.session_state.get("feature_target") if st.session_state.get("feature_target") in fe.columns else fe.columns[0]
        cols = list(fe.columns)
        tgt = st.selectbox("Target", cols, index=cols.index(default_tgt), key="tgt_model")
        features = [c for c in cols if c != tgt]

        if not features:
            st.error("No feature columns available for training.")
        else:
            model_name = st.selectbox("Model", ["LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor"], index=1)
            test_size = st.slider("Holdout Test size (%)", 10, 40, 20, step=5) / 100.0
            use_tscv = st.checkbox("Use TimeSeriesSplit CV (averages shown)", value=False)
            splits = st.slider("TimeSeriesSplit folds", 3, 10, 5) if use_tscv else None
            n_estimators = st.slider("n_estimators (RF/GB)", 50, 500, 200, step=50)
            max_depth = st.slider("max_depth (RF only, 0=None)", 0, 30, 10, step=2)
            max_depth = None if max_depth == 0 else max_depth
            lr_gb = st.slider("learning_rate (GB)", 0.01, 0.5, 0.1, step=0.01)

            def make_estimator():
                if model_name == "LinearRegression":
                    return LinearRegression()
                if model_name == "RandomForestRegressor":
                    return RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=42,
                        n_jobs=-1,
                    )
                return GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    learning_rate=lr_gb,
                    random_state=42,
                )

            if st.button("🎯 Train"):
                try:
                    X = fe[features]
                    y = pd.to_numeric(fe[tgt], errors="coerce").values

                    split_idx = int(len(X) * (1 - test_size))
                    if split_idx <= 0 or split_idx >= len(X):
                        raise ValueError("Invalid train/test split. Adjust the test size.")

                    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]

                    pipe = Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", make_estimator()),
                    ])

                    if use_tscv:
                        tscv = TimeSeriesSplit(n_splits=splits)
                        maes, mses, r2s = [], [], []
                        for tr, te in tscv.split(X):
                            pipe.fit(X.iloc[tr], y[tr])
                            pr = pipe.predict(X.iloc[te])
                            maes.append(mean_absolute_error(y[te], pr))
                            mses.append(mean_squared_error(y[te], pr))
                            r2s.append(r2_score(y[te], pr))
                        st.info(f"CV (TS split={splits}) — MAE={np.mean(maes):.4f}, MSE={np.mean(mses):.4f}, R²={np.mean(r2s):.4f}")

                    pipe.fit(X_train, y_train)
                    pred = pipe.predict(X_test)
                    mae = mean_absolute_error(y_test, pred)
                    mse = mean_squared_error(y_test, pred)
                    r2 = r2_score(y_test, pred)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("MAE", f"{mae:.4f}")
                    with c2:
                        st.metric("MSE", f"{mse:.4f}")
                    with c3:
                        st.metric("R²", f"{r2:.4f}")

                    fig, ax = plt.subplots()
                    ax.plot(y_test, label="Actual")
                    ax.plot(pred, label="Predicted", linestyle="--")
                    ax.legend()
                    ax.set_title("Test segment")
                    st.pyplot(fig)

                    fig2, ax2 = plt.subplots()
                    ax2.plot(y_test - pred)
                    ax2.set_title("Residuals")
                    st.pyplot(fig2)

                    if model_name in ["RandomForestRegressor", "GradientBoostingRegressor"]:
                        try:
                            importances = pipe.named_steps["model"].feature_importances_
                            imp_df = pd.DataFrame({"feature": features, "importance": importances}).sort_values("importance", ascending=False).head(25)
                            st.subheader("Top Feature Importances")
                            st.dataframe(imp_df)
                        except Exception:
                            pass

                    st.session_state["last_predictions"] = pd.DataFrame({"y_true": y_test, "y_pred": pred})
                except Exception as e:
                    st.error(f"Training failed: {e}")

if current == "Forecast":
    st.header("7) Forecast next N steps (recursive)")
    fe = st.session_state.get("features_df")
    if fe is None or fe.empty:
        st.warning("Please build features first.")
    else:
        default_tgt = st.session_state.get("feature_target") if st.session_state.get("feature_target") in fe.columns else fe.columns[0]
        cols = list(fe.columns)
        tgt = st.selectbox("Target", cols, index=cols.index(default_tgt), key="tgt_fore")
        features = [c for c in cols if c != tgt]

        if not features:
            st.error("No feature columns available for forecasting.")
        else:
            model_name = st.selectbox("Model", ["LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor"], index=1, key="model_fore")
            horizon = st.slider("Forecast horizon (steps)", 1, 60, 10, step=1)
            n_estimators = st.slider("n_estimators (RF/GB)", 50, 500, 200, step=50, key="nest_fore")
            max_depth = st.slider("max_depth (RF only, 0=None)", 0, 30, 10, step=2, key="md_fore")
            max_depth = None if max_depth == 0 else max_depth
            lr_gb = st.slider("learning_rate (GB)", 0.01, 0.5, 0.1, step=0.01, key="lrf_fore")

            def make_estimator():
                if model_name == "LinearRegression":
                    return LinearRegression()
                if model_name == "RandomForestRegressor":
                    return RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=42,
                        n_jobs=-1,
                    )
                return GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    learning_rate=lr_gb,
                    random_state=42,
                )

            if st.button("📈 Forecast"):
                try:
                    X = fe[features]
                    y = pd.to_numeric(fe[tgt], errors="coerce").values
                    pipe = Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", make_estimator()),
                    ]).fit(X, y)

                    future = []
                    current_row = X.iloc[-1].copy()
                    lag_cols = sorted(
                        [c for c in X.columns if c.startswith(f"{tgt}_lag")],
                        key=lambda name: int(name.split("lag")[1]),
                    )

                    for _ in range(horizon):
                        yhat = float(pipe.predict(pd.DataFrame([current_row]))[0])
                        future.append(yhat)

                        for idx, lag_col in enumerate(lag_cols):
                            if idx == 0:
                                current_row[lag_col] = yhat
                            else:
                                current_row[lag_col] = current_row[lag_cols[idx - 1]]

                    st.success(f"Generated {len(future)} forecasted steps")

                    hist_tail = min(100, len(y))
                    fig, ax = plt.subplots()
                    ax.plot(np.arange(len(y))[-hist_tail:], y[-hist_tail:], label="History (last 100)")
                    x_fore = np.arange(len(y) - 1, len(y) - 1 + len(future) + 1)
                    y_fore = np.concatenate([[y[-1]], np.asarray(future)])
                    ax.plot(x_fore, y_fore, linestyle="--", label="Forecast")
                    ax.legend()
                    ax.set_title("Recursive Forecast")
                    st.pyplot(fig)

                    fut_df = pd.DataFrame({"step": np.arange(1, len(future) + 1), "forecast": future})
                    st.session_state["future_forecast"] = fut_df
                    st.dataframe(fut_df)
                except Exception as e:
                    st.error(f"Forecast failed: {e}")

if current == "Export":
    st.header("8) Export")
    preds = st.session_state.get("last_predictions")
    fut = st.session_state.get("future_forecast")
    if preds is not None:
        st.subheader("Test Predictions")
        st.dataframe(preds.head(25))
        to_csv_download(preds, "test_predictions.csv")
    if fut is not None:
        st.subheader("Future Forecast")
        st.dataframe(fut.head(25))
        to_csv_download(fut, "future_forecast.csv")
    if preds is None and fut is None:
        st.info("Run Train/Forecast steps to export CSVs.")
