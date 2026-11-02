"""Term-deposit subscription classifier: Streamlit dashboard (inference only).

BITS Pilani M.Tech AIML, Machine Learning Assignment 2 (student 2026AJ05200).

What the app does
* Loads the five pipelines saved by ``python -m src.train`` from ``model/``.
  Nothing is (re)fitted here; uploaded data is only transformed and scored.
* Accepts a labelled test CSV (``,`` or ``;``) or the bundled ``test_data.csv``.
* Shows the six metrics, confusion matrix and classification report for the
  chosen model, compares all five models (leaderboard, bar chart, ROC curves),
  and lets the user filter and download the predictions.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import html
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve

from src.evaluate import (
    METRIC_KEYS,
    METRIC_NAMES,
    MODEL_FILES,
    comparison_table,
    evaluate_model,
    load_metadata,
    load_models,
)
from src.preprocess import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NEGATIVE_LABEL,
    NUMERICAL_FEATURES,
    POSITIVE_LABEL,
    TARGET,
    TEST_DATA_PATH,
    DataValidationError,
    clean_dataframe,
    prepare_features_and_target,
    read_csv_auto,
    validate_dataframe,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
GITHUB_URL = "https://github.com/sarathsidharth/2026aj05200-s1-ml-classification-streamlit-assignment"
STUDENT_ID = "2026AJ05200"
MAX_UPLOAD_MB = 25
UPLOAD_STORE_SIZE = 8

# Short codes keep the URL readable (?model=svm&data=sample)
MODEL_CODES = {
    "lr": "Logistic Regression",
    "dt": "Decision Tree",
    "knn": "KNN",
    "nb": "Naive Bayes",
    "svm": "Support Vector Machine",
}
CODE_OF = {v: k for k, v in MODEL_CODES.items()}
DEFAULT_MODEL = "svm"
SOURCES = {"upload": "Upload CSV", "sample": "Use sample"}

MODEL_COLORS = ["#2563eb", "#f97316", "#10b981", "#a855f7", "#e11d48"]
KEY_PARAMS = {
    "Logistic Regression": ["C", "penalty", "solver", "max_iter", "class_weight"],
    "Decision Tree": ["criterion", "max_depth", "min_samples_leaf", "class_weight"],
    "KNN": ["n_neighbors", "weights", "metric", "p"],
    "Naive Bayes": ["var_smoothing", "priors"],
    "Support Vector Machine": ["kernel", "C", "gamma", "class_weight"],
}
METRIC_HELP = {
    "Accuracy": "Share of all clients classified correctly. Misleading under imbalance: always 'no' scores ~88%.",
    "AUC": "Area under the ROC curve: how well the scores rank subscribers above non-subscribers (0.5 = random).",
    "Precision": "Of the clients predicted 'yes', the share who really subscribed.",
    "Recall": "Of the clients who really subscribed, the share the model found.",
    "F1": "Harmonic mean of precision and recall for the 'yes' class.",
    "MCC": "Matthews correlation: balanced quality score from all four confusion-matrix cells (-1 to 1, 0 = random).",
}

st.set_page_config(
    page_title=f"Term Deposit Classifier · {STUDENT_ID}",
    page_icon=":material/savings:",
    layout="wide",
    initial_sidebar_state="auto",  # collapsed on phones so it does not cover the page
)

# --------------------------------------------------------------------------- #
# Styling: theme-neutral (inherits text colour, translucent fills) so it
# works with both the light and the dark theme defined in .streamlit/config.toml
# --------------------------------------------------------------------------- #
CSS = """
<style>
.block-container {padding-top: 4.2rem; padding-bottom: 3rem; max-width: 1280px;}
.td-hero {
  position: relative; border-radius: 18px; padding: 1.6rem 1.8rem 1.4rem;
  color: #f8fafc; overflow: hidden;
  background: radial-gradient(1200px 300px at 85% -40%, rgba(96,165,250,.35), transparent 60%),
              linear-gradient(120deg, #0b1d3f 0%, #12306b 55%, #1d4ed8 120%);
  box-shadow: 0 10px 30px rgba(15,23,42,.25);
}
.td-hero .td-kicker {font-size: .74rem; letter-spacing: .12em; text-transform: uppercase; color: #bfdbfe; font-weight: 600;}
.td-hero h1 {color: #fff; font-size: 2.05rem; line-height: 1.15; margin: .35rem 0 .45rem; padding: 0; font-weight: 800;}
.td-hero p.td-lead {color: #dbe6fb; max-width: 760px; margin: 0; font-size: .98rem;}
.td-stats {display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: .7rem; margin-top: 1.2rem;}
.td-stat {background: rgba(255,255,255,.09); border: 1px solid rgba(255,255,255,.16); border-radius: 12px; padding: .65rem .85rem;}
.td-stat b {display: block; font-size: 1.35rem; font-weight: 800; color: #fff;}
.td-stat span {font-size: .76rem; color: #c7d6f3;}
.td-theme {
  position: absolute; top: 1.1rem; right: 1.2rem; display: inline-flex; align-items: center; gap: .45rem;
  background: rgba(255,255,255,.12); color: #fff; border: 1px solid rgba(255,255,255,.3);
  border-radius: 999px; padding: .38rem .85rem; font: 600 .82rem Inter, sans-serif; cursor: pointer;
}
.td-theme:hover {background: rgba(255,255,255,.22);}
.td-theme:focus-visible {outline: 2px solid #93c5fd; outline-offset: 2px;}
.td-theme .td-ico {font-size: 1rem; line-height: 1;}

.td-card {border: 1px solid rgba(127,127,127,.24); background: rgba(127,127,127,.05);
          border-radius: 14px; padding: 1rem 1.1rem; height: 100%;}
.td-card h4 {margin: 0 0 .35rem; font-size: 1rem; padding: 0;}
.td-card p {margin: 0; font-size: .9rem; opacity: .85;}
.td-step {font-size: .72rem; font-weight: 700; letter-spacing: .08em; color: #2563eb; text-transform: uppercase;}

.td-metrics {display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: .7rem; margin: .4rem 0 1rem;}
.td-metric {border: 1px solid rgba(127,127,127,.24); border-radius: 14px; padding: .8rem .9rem;
            background: rgba(127,127,127,.05); border-top: 3px solid #2563eb;}
.td-metric .lbl {font-size: .78rem; font-weight: 600; opacity: .75; text-transform: uppercase; letter-spacing: .05em;}
.td-metric .val {font-size: 1.6rem; font-weight: 800; margin: .1rem 0;}
.td-metric .ref {font-size: .74rem; opacity: .7;}
.td-metric .up {color: #16a34a;} .td-metric .down {color: #dc2626;}

.td-insight {border-left: 4px solid #2563eb; background: rgba(37,99,235,.08); border-radius: 0 12px 12px 0;
             padding: .8rem 1rem; margin-bottom: 1rem; font-size: .93rem;}
.td-insight b {font-weight: 700;}
.td-winner {border-radius: 16px; padding: 1.1rem 1.3rem; color: #fff; margin-bottom: 1rem;
            background: linear-gradient(120deg, #0f2a5c, #1d4ed8);}
.td-winner .t {font-size: .74rem; letter-spacing: .1em; text-transform: uppercase; color: #bfdbfe; font-weight: 700;}
.td-winner .n {font-size: 1.5rem; font-weight: 800; margin: .15rem 0 .3rem;}
.td-winner p {margin: 0; color: #dbe6fb; font-size: .9rem;}
.td-pill {display: inline-block; font-size: .72rem; font-weight: 700; padding: .12rem .55rem; border-radius: 999px;
          background: rgba(37,99,235,.14); color: #2563eb; margin-right: .3rem;}
.td-params {width: 100%; font-size: .82rem; border-collapse: collapse; margin-top: .5rem;}
.td-params td {padding: .22rem 0; border-bottom: 1px dashed rgba(127,127,127,.25);}
.td-params td:last-child {text-align: right; font-family: ui-monospace, monospace;}
.td-side-h {font-size: .72rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; opacity: .65; margin: .6rem 0 .1rem;}
.td-brand {display: flex; gap: .6rem; align-items: center; margin-bottom: .3rem;}
.td-brand .m {width: 38px; height: 38px; border-radius: 10px; display: grid; place-items: center; font-weight: 800;
              color: #fff; background: linear-gradient(135deg, #12306b, #2563eb);}
.td-brand .t {font-weight: 800; line-height: 1.1;}
.td-brand .s {font-size: .75rem; opacity: .7;}
.td-foot {font-size: .78rem; opacity: .65; text-align: center; margin-top: 2rem;}

@media (max-width: 900px) {
  .td-metrics {grid-template-columns: repeat(3, minmax(0, 1fr));}
  .td-stats {grid-template-columns: repeat(3, minmax(0, 1fr));}
}
@media (max-width: 640px) {
  .block-container {padding-left: 1rem; padding-right: 1rem; padding-top: 3.6rem;}
  .td-hero {padding: 1.2rem 1.1rem 1.1rem; border-radius: 14px;}
  .td-hero h1 {font-size: 1.45rem; padding-right: 2.6rem;}
  .td-hero p.td-lead {font-size: .9rem;}
  .td-stats {grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .5rem;}
  .td-stat b {font-size: 1.1rem;}
  .td-theme {top: .9rem; right: .9rem; padding: .4rem .55rem;}
  .td-theme .td-lbl {display: none;}
  .td-metrics {grid-template-columns: repeat(2, minmax(0, 1fr));}
  .td-metric .val {font-size: 1.3rem;}
}
</style>
"""

# Light/dark switch in the header. It stores the choice where Streamlit's own
# Settings menu does (localStorage "stActiveTheme-<path>-v2"), then reloads.
# The model and data source live in the URL (bind="query-params") and an upload
# is restored from the server-side store via ?file=, so nothing is lost.
# (st.html strips inline onclick handlers, so the listener is attached in JS.)
THEME_JS = r"""
<script>
(function () {
  const KEY = "stActiveTheme-" + window.location.pathname + "-v2";
  function isDark() {
    const el = document.querySelector('[data-testid="stApp"]') || document.body;
    const rgb = (getComputedStyle(el).backgroundColor.match(/[\d.]+/g) || []).map(Number);
    if (rgb.length < 3) return window.matchMedia("(prefers-color-scheme: dark)").matches;
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] < 128;
  }
  function sync() {
    document.querySelectorAll(".td-theme").forEach(function (btn) {
      const dark = isDark();
      btn.querySelector(".td-ico").textContent = dark ? "☀" : "☾";
      btn.querySelector(".td-lbl").textContent = dark ? "Light mode" : "Dark mode";
      btn.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
      btn.onclick = function () {
        try { window.localStorage.setItem(KEY, JSON.stringify(isDark() ? "Light" : "Dark")); } catch (e) {}
        window.location.reload();
      };
    });
  }
  sync();
  [150, 600, 1500, 3000].forEach(function (ms) { setTimeout(sync, ms); });
})();
</script>
"""


def esc(value) -> str:
    return html.escape(str(value))


def fmt(value: float, digits: int = 4) -> str:
    return "n/a" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.{digits}f}"


# --------------------------------------------------------------------------- #
# Cached resources (loaded once per server process)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the five trained models…")
def get_models() -> dict:
    return load_models()


@st.cache_data(show_spinner=False)
def get_metadata() -> dict:
    return load_metadata()


@st.cache_data(show_spinner=False)
def sample_bytes() -> bytes:
    return TEST_DATA_PATH.read_bytes()


@st.cache_resource
def upload_store() -> dict:
    """token -> (file name, bytes) for recent uploads, so an upload survives the
    page reload done by the theme switch. Holds at most UPLOAD_STORE_SIZE files."""
    return {}


def file_token(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:20]


@st.cache_data(show_spinner=False, max_entries=16)
def parse_upload(token: str, _data: bytes) -> tuple[pd.DataFrame | None, list[str]]:
    """Read + validate the CSV. Cached per file hash (``token``)."""
    try:
        df = read_csv_auto(_data)
    except DataValidationError as exc:
        return None, exc.errors
    except Exception as exc:  # malformed CSV, wrong encoding, ...
        return None, [f"Could not read the file as CSV ({type(exc).__name__}: {exc})."]
    return df, validate_dataframe(df, require_target=True)


@st.cache_data(show_spinner=False, max_entries=8)
def score_all_models(token: str, _df: pd.DataFrame) -> dict:
    """Score all five saved models on the uploaded data, once per file.

    SVM inference on ~9k rows takes ~10 s, so results are cached by file hash.
    Only transform/predict is called; no model is fitted on uploaded data.
    """
    X, y = prepare_features_and_target(_df)
    out = {}
    for name, model in get_models().items():
        t0 = time.perf_counter()
        res = evaluate_model(model, X, y)
        res["seconds"] = time.perf_counter() - t0
        if y.nunique() == 2:
            fpr, tpr, _ = roc_curve(y, res["scores"])
            if len(fpr) > 400:  # thin the curve for plotting; keep both ends
                idx = np.unique(np.linspace(0, len(fpr) - 1, 400).astype(int))
                fpr, tpr = fpr[idx], tpr[idx]
            res["roc"] = (fpr, tpr)
        else:
            res["roc"] = None
        out[name] = res
    return {"results": out, "y": y.to_numpy()}


# --------------------------------------------------------------------------- #
# Load artifacts
# --------------------------------------------------------------------------- #
try:
    metadata = get_metadata()
    get_models()
except FileNotFoundError as exc:
    st.error(f"Saved models are missing: {exc}")
    st.stop()

ds_meta = metadata["dataset"]
split_meta = metadata["split"]
model_meta = metadata["models"]

st.html(CSS)

# --------------------------------------------------------------------------- #
# Sidebar: controls
# --------------------------------------------------------------------------- #
def on_upload_change() -> None:
    # The user removed the file with the uploader's x: forget the stored token
    if st.session_state.get("uploader") is None:
        st.query_params.pop("file", None)


def forget_restored() -> None:
    st.query_params.pop("file", None)


def use_sample() -> None:
    st.session_state["data"] = "sample"


with st.sidebar:
    st.html(
        '<div class="td-brand"><div class="m">TD</div><div>'
        '<div class="t">Term Deposit Classifier</div>'
        f'<div class="s">ML Assignment 2 · {STUDENT_ID}</div></div></div>'
    )

    st.html('<div class="td-side-h">1 · Test data</div>')
    # Initialise from the URL ourselves (instead of `default=`) because the
    # welcome-screen button also sets this value through session state.
    if "data" not in st.session_state:
        # (bound widgets write their display label to the URL, e.g. ?data=Use+sample)
        wanted = st.query_params.get("data")
        by_label = {label: code for code, label in SOURCES.items()}
        st.session_state["data"] = wanted if wanted in SOURCES else by_label.get(wanted, "upload")
    source = st.segmented_control(
        "Test data source", options=list(SOURCES), format_func=SOURCES.get,
        required=True, key="data", bind="query-params",
        label_visibility="collapsed", width="stretch",
    ) or "upload"

    data_bytes, data_name, restored = None, None, False
    if source == "upload":
        uploaded = st.file_uploader(
            "Upload a labelled test CSV", type=["csv"], key="uploader",
            max_upload_size=MAX_UPLOAD_MB, on_change=on_upload_change,
            help=(f"Comma- or semicolon-separated. Needs the {len(FEATURES)} input columns "
                  f"and the target column '{TARGET}' with yes/no values. Extra columns "
                  "(e.g. 'duration') are ignored."),
        )
        store = upload_store()
        if uploaded is not None:
            data_bytes, data_name = uploaded.getvalue(), uploaded.name
            token = file_token(data_bytes)
            store[token] = (data_name, data_bytes)
            while len(store) > UPLOAD_STORE_SIZE:
                store.pop(next(iter(store)))
            if st.query_params.get("file") != token:
                st.query_params["file"] = token
        else:
            token = st.query_params.get("file")
            if token and token in store:
                data_name, data_bytes = store[token]
                restored = True
                st.info(f"Using **{esc(data_name)}** (kept after reload).", icon=":material/history:")
                st.button("Remove this file", on_click=forget_restored, type="tertiary",
                          icon=":material/close:")
            elif token:
                st.query_params.pop("file", None)
    else:
        data_bytes, data_name = sample_bytes(), "test_data.csv (bundled)"
        st.caption(f"The held-out 20% split: {split_meta['test']['rows']:,} rows never used for training.")

    st.download_button(
        "Download sample test_data.csv", data=sample_bytes(), file_name="test_data.csv",
        mime="text/csv", type="tertiary", icon=":material/download:", on_click="ignore",
    )

    st.html('<div class="td-side-h">2 · Model</div>')
    model_code = st.selectbox(
        "Choose a model", options=list(MODEL_CODES), format_func=MODEL_CODES.get,
        index=list(MODEL_CODES).index(DEFAULT_MODEL), key="model", bind="query-params",
        help="All five models were trained on the same 80% split with the same preprocessing.",
    )
    model_name = MODEL_CODES[model_code or DEFAULT_MODEL]
    info = model_meta[model_name]
    rows = "".join(
        f"<tr><td>{esc(p)}</td><td>{esc(info['params'].get(p))}</td></tr>" for p in KEY_PARAMS[model_name]
    )
    st.html(
        f'<div class="td-card"><span class="td-pill">{esc(info["estimator"])}</span>'
        f'<span class="td-pill">{esc(info["score_source"])}</span>'
        f'<table class="td-params">{rows}</table></div>'
    )
    with st.expander("Why these settings?"):
        st.write(info["rationale"])

    st.html('<div class="td-side-h">About</div>')
    st.caption("BITS Pilani · M.Tech AIML · Machine Learning · Assignment 2. "
               "Dataset: UCI Bank Marketing (bank-full).")
    st.link_button("Source code on GitHub", GITHUB_URL, icon=":material/code:", width="stretch")

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.html(
    f"""
    <div class="td-hero">
      <button class="td-theme" type="button" aria-label="Toggle light or dark mode">
        <span class="td-ico">&#9790;</span><span class="td-lbl">Dark mode</span>
      </button>
      <div class="td-kicker">BITS Pilani · M.Tech AIML · ML Assignment 2 · {STUDENT_ID}</div>
      <h1>Term Deposit Subscription Classifier</h1>
      <p class="td-lead">Will a bank client subscribe to a term deposit after a marketing call?
      Upload labelled test data and compare five trained classifiers on it, using
      metrics that stay honest under an 88/12 class imbalance.</p>
      <div class="td-stats">
        <div class="td-stat"><b>{ds_meta['rows']:,}</b><span>clients in UCI dataset</span></div>
        <div class="td-stat"><b>{ds_meta['model_input_features']}</b><span>input features (duration excluded)</span></div>
        <div class="td-stat"><b>5</b><span>trained classifiers</span></div>
        <div class="td-stat"><b>6</b><span>evaluation metrics</span></div>
        <div class="td-stat"><b>{split_meta['test']['rows']:,}</b><span>held-out test rows</span></div>
      </div>
    </div>
    {THEME_JS}
    """,
    unsafe_allow_javascript=True,
)

# --------------------------------------------------------------------------- #
# Welcome screen (no data yet)
# --------------------------------------------------------------------------- #
if data_bytes is None:
    st.write("")
    st.subheader("Get started in three steps")
    c1, c2, c3 = st.columns(3, gap="medium")
    steps = [
        ("Step 1", "Choose test data",
         "Upload a labelled CSV in the sidebar (comma or semicolon separated), or use the bundled "
         f"held-out set of {split_meta['test']['rows']:,} clients."),
        ("Step 2", "Pick a model",
         "Logistic Regression, Decision Tree, KNN, Naive Bayes or SVM. All were trained once on "
         "the same 80% split and are only loaded here, never retrained."),
        ("Step 3", "Read the results",
         "Six metrics, a confusion matrix and classification report, a five-model comparison "
         "with ROC curves, and downloadable predictions."),
    ]
    for col, (step, title, text) in zip((c1, c2, c3), steps):
        col.html(f'<div class="td-card"><div class="td-step">{step}</div><h4>{title}</h4><p>{text}</p></div>')
    st.write("")
    st.button("Evaluate on the held-out test set", type="primary", icon=":material/play_arrow:",
              on_click=use_sample)
    with st.expander("What must an uploaded CSV contain?"):
        st.markdown(
            f"- **Input features ({len(FEATURES)}):** numerical `{', '.join(NUMERICAL_FEATURES)}`; "
            f"categorical `{', '.join(CATEGORICAL_FEATURES)}`.\n"
            f"- **Target:** `{TARGET}` with values `{POSITIVE_LABEL}` / `{NEGATIVE_LABEL}` (positive class = `{POSITIVE_LABEL}`).\n"
            "- Comma- or semicolon-separated; header names are case-insensitive; extra columns are ignored.\n"
            "- Easiest option: download the sample `test_data.csv` from the sidebar and upload it."
        )
    st.html(f'<div class="td-foot">Inference only: models are loaded from <code>model/*.pkl</code>. '
            f'Student {STUDENT_ID}.</div>')
    st.stop()

# --------------------------------------------------------------------------- #
# Validate and score
# --------------------------------------------------------------------------- #
token = file_token(data_bytes)
raw_df, errors = parse_upload(token, data_bytes)
if errors:
    st.error(f"**{esc(data_name)}** can't be evaluated:", icon=":material/error:")
    for err in errors:
        st.markdown(f"- {err}")
    st.info("Tip: download the sample `test_data.csv` from the sidebar to see the expected format.",
            icon=":material/lightbulb:")
    st.stop()

clean_df = clean_dataframe(raw_df)
n_rows = len(clean_df)
n_blank = int(clean_df[FEATURES].isna().sum().sum())
# All five pipelines share the same training categories; read them from one encoder
encoder = (get_models()[model_name].named_steps["preprocessor"]
           .named_transformers_["cat"].named_steps["onehot"])
unseen = {}
for col, cats in zip(CATEGORICAL_FEATURES, encoder.categories_):
    new_vals = sorted(set(clean_df[col].dropna().unique()) - set(cats))
    if new_vals:
        unseen[col] = new_vals

with st.spinner(f"Scoring all five models on {n_rows:,} rows (the SVM takes a few seconds)…"):
    scored = score_all_models(token, clean_df)
results, y_true = scored["results"], scored["y"]
res = results[model_name]
m, cm = res["metrics"], res["confusion_matrix"]
n_pos = int(y_true.sum())
single_class = len(np.unique(y_true)) < 2
is_reference = token == file_token(sample_bytes())

notes = []
if n_blank:
    notes.append(f"{n_blank:,} blank feature cell(s) were filled by the imputers learned on the training data.")
if unseen:
    notes.append("Categories not seen in training were ignored by the encoder: "
                 + "; ".join(f"{c}: {', '.join(map(str, v[:4]))}" for c, v in unseen.items()) + ".")
if single_class:
    notes.append("The target has only one class, so AUC is undefined and MCC is 0.")
for note in notes:
    st.warning(note, icon=":material/info:")

st.html(
    f'<div style="margin:.9rem 0 .2rem;font-size:.95rem">Evaluating <b>{esc(model_name)}</b> on '
    f'<b>{esc(data_name)}</b> · {n_rows:,} rows · {n_pos:,} actual “{POSITIVE_LABEL}” '
    f'({n_pos / n_rows:.1%})' + (' · <span class="td-pill">bundled held-out set</span>' if is_reference else "")
    + "</div>"
)

tab_eval, tab_cmp, tab_pred, tab_info = st.tabs(
    [":material/analytics: Evaluation", ":material/leaderboard: Model comparison",
     ":material/table_view: Predictions", ":material/dataset: Dataset & models"]
)

# --------------------------------------------------------------------------- #
# Tab 1: evaluation of the selected model
# --------------------------------------------------------------------------- #
with tab_eval:
    ref = model_meta[model_name]["test_metrics"]
    cards = []
    for name in METRIC_NAMES:
        key = METRIC_KEYS[name]
        val, ref_val = m[key], ref[key]
        if is_reference:
            ref_txt = "matches training-time result ✓" if abs(val - ref_val) < 1e-9 else f"reference {fmt(ref_val)}"
        elif np.isnan(val):
            ref_txt = f"reference {fmt(ref_val)}"
        else:
            diff = val - ref_val
            cls = "up" if diff >= 0 else "down"
            ref_txt = f'reference {fmt(ref_val)} · <span class="{cls}">{diff:+.4f}</span>'
        cards.append(f'<div class="td-metric" title="{esc(METRIC_HELP[name])}"><div class="lbl">{name}</div>'
                     f'<div class="val">{fmt(val)}</div><div class="ref">{ref_txt}</div></div>')
    st.html(f'<div class="td-metrics">{"".join(cards)}</div>')
    st.caption("Reference = the same model's score on the held-out split at training time (model/metadata.json). "
               "Hover a card for what the metric means.")

    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    baseline = max(n_pos, n_rows - n_pos) / n_rows
    flagged = tp + fp
    insight = [
        f"Of the <b>{n_pos:,}</b> clients who actually subscribed, {esc(model_name)} found "
        f"<b>{tp:,}</b> (recall {m['recall']:.1%}) and missed {fn:,}.",
        f"It flagged <b>{flagged:,}</b> clients as likely subscribers; <b>{tp:,}</b> of them really did "
        f"(precision {m['precision']:.1%}), so {fp:,} calls would go to non-subscribers.",
        f"Accuracy is {m['accuracy']:.1%}, against {baseline:.1%} for simply predicting the majority class "
        "every time, which is why MCC, F1 and AUC are the fairer yardsticks here.",
    ]
    st.html('<div class="td-insight">' + "<br>".join(insight) + "</div>")

    c_left, c_right = st.columns([1, 1.15], gap="large")
    with c_left:
        st.markdown("##### Confusion matrix")
        cm_df = pd.DataFrame([
            {"Actual": NEGATIVE_LABEL, "Predicted": NEGATIVE_LABEL, "n": tn, "cell": "TN"},
            {"Actual": NEGATIVE_LABEL, "Predicted": POSITIVE_LABEL, "n": fp, "cell": "FP"},
            {"Actual": POSITIVE_LABEL, "Predicted": NEGATIVE_LABEL, "n": fn, "cell": "FN"},
            {"Actual": POSITIVE_LABEL, "Predicted": POSITIVE_LABEL, "n": tp, "cell": "TP"},
        ])
        row_tot = cm_df.groupby("Actual")["n"].transform("sum").replace(0, 1)
        cm_df["share"] = cm_df["n"] / row_tot
        cm_df["text"] = cm_df.apply(lambda r: f"{r['cell']}  {r['n']:,}", axis=1)
        cm_df["sub"] = cm_df["share"].map(lambda s: f"{s:.1%} of actual")
        order = [NEGATIVE_LABEL, POSITIVE_LABEL]
        base = alt.Chart(cm_df).encode(
            x=alt.X("Predicted:N", sort=order, axis=alt.Axis(labelAngle=0, orient="top")),
            y=alt.Y("Actual:N", sort=order),
        )
        heat = base.mark_rect(cornerRadius=6).encode(
            color=alt.Color("share:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]), legend=None),
            tooltip=["Actual", "Predicted", "cell", alt.Tooltip("n:Q", format=","), alt.Tooltip("share:Q", format=".1%")],
        )
        txt_color = alt.condition(alt.datum.share > 0.55, alt.value("#ffffff"), alt.value("#0f172a"))
        label = base.mark_text(fontSize=17, fontWeight="bold", dy=-8).encode(text="text:N", color=txt_color)
        sub = base.mark_text(fontSize=11, dy=12).encode(text="sub:N", color=txt_color)
        st.altair_chart((heat + label + sub).properties(height=290), width="stretch")
        st.caption("Rows = actual class, columns = predicted class; shading = share of the actual row.")

    with c_right:
        st.markdown("##### Classification report")
        rep = res["report"].copy()
        rep.index = [f"{i} (positive)" if i == POSITIVE_LABEL else i for i in rep.index]
        shown = pd.DataFrame(index=rep.index)
        for col in ["precision", "recall", "f1-score"]:
            shown[col] = rep[col].map(lambda v: f"{v:.4f}")
        shown["support"] = rep["support"].map(lambda v: f"{int(round(v)):,}")
        if "accuracy" in shown.index:  # sklearn repeats accuracy in every column; show it once
            shown.loc["accuracy", ["precision", "recall"]] = ""
            shown.loc["accuracy", "support"] = f"{n_rows:,}"
        st.dataframe(shown, width="stretch")
        st.caption(f"Positive class = “{POSITIVE_LABEL}”. Scores: {res['score_source']} "
                   f"(scored in {res['seconds']:.1f}s).")

# --------------------------------------------------------------------------- #
# Tab 2: compare all five models on the same data
# --------------------------------------------------------------------------- #
with tab_cmp:
    table = comparison_table(results, decimals=None)
    ranked = table.sort_values(["MCC", "F1", "AUC"], ascending=False)
    winner = ranked.index[0]
    best_acc = table["Accuracy"].idxmax()
    st.html(
        f'<div class="td-winner"><div class="t">Best overall on this data (ranked by MCC, then F1, AUC)</div>'
        f'<div class="n">{esc(winner)}</div>'
        f'<p>MCC {fmt(table.loc[winner, "MCC"])} · F1 {fmt(table.loc[winner, "F1"])} · '
        f'AUC {fmt(table.loc[winner, "AUC"])} · Recall {fmt(table.loc[winner, "Recall"])}. '
        f'Highest accuracy: {esc(best_acc)} ({fmt(table.loc[best_acc, "Accuracy"])})'
        + ("; but accuracy rewards predicting the majority class, so it is not used to pick the winner."
           if best_acc != winner else ".")
        + "</p></div>"
    )

    st.markdown("##### Leaderboard")
    board = ranked.reset_index()
    board.insert(0, "Rank", range(1, len(board) + 1))
    styled = (board.style.format({c: "{:.4f}" for c in METRIC_NAMES})
              .highlight_max(subset=METRIC_NAMES, props="font-weight:700; background-color: rgba(37,99,235,.18)"))
    st.dataframe(styled, width="stretch", hide_index=True, column_config={
        "Rank": st.column_config.NumberColumn("Rank", width="small"),
        "ML Model Name": st.column_config.TextColumn("ML Model Name", width="medium"),
    })
    st.caption("Bold/shaded = best value per metric. All models scored on the same rows.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("##### Metrics by model")
        long = table.reset_index().melt(id_vars="ML Model Name", var_name="Metric", value_name="Score")
        bars = alt.Chart(long).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("Metric:N", sort=METRIC_NAMES, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("ML Model Name:N", sort=list(MODEL_FILES)),
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1]), title="Score"),
            color=alt.Color("ML Model Name:N", sort=list(MODEL_FILES), title=None,
                            scale=alt.Scale(domain=list(MODEL_FILES), range=MODEL_COLORS),
                            legend=alt.Legend(orient="bottom", columns=3, labelLimit=0)),
            tooltip=["ML Model Name", "Metric", alt.Tooltip("Score:Q", format=".4f")],
        ).properties(height=360)
        st.altair_chart(bars, width="stretch")
    with c2:
        st.markdown("##### ROC curves")
        if single_class:
            st.info("ROC curves need both classes in the target column.")
        else:
            roc_rows = []
            for name, r in results.items():
                fpr, tpr = r["roc"]
                label = f"{name} (AUC {r['metrics']['auc']:.3f})"
                roc_rows += [{"Model": label, "FPR": a, "TPR": b, "order": i} for i, (a, b) in enumerate(zip(fpr, tpr))]
            roc_df = pd.DataFrame(roc_rows)
            labels = [f"{n} (AUC {results[n]['metrics']['auc']:.3f})" for n in MODEL_FILES]
            lines = alt.Chart(roc_df).mark_line(strokeWidth=2).encode(
                x=alt.X("FPR:Q", title="False positive rate", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("TPR:Q", title="True positive rate (recall)", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Model:N", sort=labels, title=None,
                                scale=alt.Scale(domain=labels, range=MODEL_COLORS),
                                legend=alt.Legend(orient="bottom-right", labelLimit=0, padding=6)),
                order="order:Q",
                tooltip=["Model", alt.Tooltip("FPR:Q", format=".3f"), alt.Tooltip("TPR:Q", format=".3f")],
            )
            diag = alt.Chart(pd.DataFrame({"FPR": [0, 1], "TPR": [0, 1]})).mark_line(
                strokeDash=[5, 5], color="#94a3b8").encode(x="FPR:Q", y="TPR:Q")
            st.altair_chart((diag + lines).properties(height=360), width="stretch")

    with st.expander("Training reference metrics (from model/metadata.json)"):
        ref_test = comparison_table({n: model_meta[n]["test_metrics"] for n in MODEL_FILES}, decimals=4)
        ref_train = comparison_table({n: model_meta[n]["train_metrics"] for n in MODEL_FILES}, decimals=4)
        st.markdown(f"**Held-out test split** ({split_meta['test']['rows']:,} rows), computed at training time:")
        st.dataframe(ref_test, width="stretch")
        st.markdown(f"**Training split** ({split_meta['train']['rows']:,} rows): a large train–test gap means overfitting:")
        st.dataframe(ref_train, width="stretch")

# --------------------------------------------------------------------------- #
# Tab 3: predictions
# --------------------------------------------------------------------------- #
with tab_pred:
    score_col = f"P({POSITIVE_LABEL})" if res["score_source"] == "predict_proba" else "SVM score"
    pred = raw_df.copy()
    pred.insert(0, "predicted", np.where(res["y_pred"] == 1, POSITIVE_LABEL, NEGATIVE_LABEL))
    pred.insert(1, score_col, np.round(res["scores"], 4))
    actual = np.where(y_true == 1, POSITIVE_LABEL, NEGATIVE_LABEL)
    outcome = np.select(
        [(y_true == 1) & (res["y_pred"] == 1), (y_true == 0) & (res["y_pred"] == 0),
         (y_true == 0) & (res["y_pred"] == 1)],
        ["TP", "TN", "FP"], default="FN",
    )
    pred.insert(2, "actual", actual)
    pred.insert(3, "outcome", outcome)
    pred = pred.drop(columns=[TARGET])

    filters = {
        "All": np.ones(len(pred), dtype=bool),
        "Predicted yes": pred["predicted"].eq(POSITIVE_LABEL).to_numpy(),
        "Correct": np.isin(outcome, ["TP", "TN"]),
        "Missed subscribers (FN)": outcome == "FN",
        "False alarms (FP)": outcome == "FP",
    }
    choice = st.pills("Show", list(filters), default="All", key="pred_filter") or "All"
    view = pred[filters[choice]].sort_values(score_col, ascending=False)
    st.caption(f"{len(view):,} of {len(pred):,} rows · sorted by {score_col}, most likely subscribers first")
    st.dataframe(
        view, width="stretch", height=430, hide_index=True,
        column_config={
            score_col: st.column_config.NumberColumn(score_col, format="%.4f", help=(
                "Signed distance from the SVM decision boundary; > 0 means 'yes'." if score_col == "SVM score"
                else "Predicted probability of 'yes'; > 0.5 means 'yes'.")),
            "outcome": st.column_config.TextColumn("outcome", help="TP/TN = correct, FP = false alarm, FN = missed subscriber"),
        },
    )
    d1, d2 = st.columns(2)
    slug = CODE_OF[model_name]
    d1.download_button(f"Download these {len(view):,} rows (CSV)", view.to_csv(index=False).encode(),
                       file_name=f"predictions_{slug}_{choice.split(' (')[0].lower().replace(' ', '_')}.csv",
                       mime="text/csv", icon=":material/download:", on_click="ignore", width="stretch")
    d2.download_button(f"Download all {len(pred):,} predictions (CSV)", pred.to_csv(index=False).encode(),
                       file_name=f"predictions_{slug}_all.csv", mime="text/csv", type="primary",
                       icon=":material/download:", on_click="ignore", width="stretch")

# --------------------------------------------------------------------------- #
# Tab 4: dataset and model information
# --------------------------------------------------------------------------- #
with tab_info:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("##### Uploaded data")
        st.markdown(
            f"- **File:** {esc(data_name)}\n"
            f"- **Rows × columns:** {n_rows:,} × {raw_df.shape[1]}\n"
            f"- **Target:** {n_pos:,} {POSITIVE_LABEL} / {n_rows - n_pos:,} {NEGATIVE_LABEL} "
            f"({n_pos / n_rows:.1%} positive)\n"
            f"- **Blank feature cells:** {n_blank:,}\n"
            f"- **Extra columns (ignored):** {', '.join(c for c in raw_df.columns if c not in FEATURES + [TARGET]) or 'none'}"
        )
        st.dataframe(raw_df.head(8), width="stretch", hide_index=True)
    with c2:
        st.markdown("##### Training dataset")
        cd = ds_meta["class_distribution"]
        excl = metadata["excluded_features"]
        st.markdown(
            f"- **Dataset:** [{ds_meta['name']}]({ds_meta['source']})\n"
            f"- **Size:** {ds_meta['rows']:,} rows, {ds_meta['original_input_features']} original input features\n"
            f"- **Target `{TARGET}`:** {cd[POSITIVE_LABEL]:,} {POSITIVE_LABEL} ({cd['positive_rate']:.1%}) / "
            f"{cd[NEGATIVE_LABEL]:,} {NEGATIVE_LABEL}\n"
            f"- **Split:** stratified {1 - split_meta['test_size']:.0%}/{split_meta['test_size']:.0%}, "
            f"random_state={split_meta['random_state']} → {split_meta['train']['rows']:,} train / "
            f"{split_meta['test']['rows']:,} test\n"
            f"- **Model inputs:** {len(FEATURES)} features → {metadata['preprocessing']['output_columns']} "
            "columns after one-hot encoding"
        )
        for feat, why in excl.items():
            st.info(f"**`{feat}` excluded:** {why}", icon=":material/block:")
        feat_df = pd.DataFrame(
            [{"feature": f, "type": "numerical", "preprocessing": "median impute → StandardScaler"} for f in NUMERICAL_FEATURES]
            + [{"feature": f, "type": "categorical", "preprocessing": "most-frequent impute → one-hot"} for f in CATEGORICAL_FEATURES]
        )
        st.dataframe(feat_df, width="stretch", hide_index=True, height=250)

    st.markdown("##### The five models")
    model_rows = []
    for name in MODEL_FILES:
        mm = model_meta[name]
        model_rows.append({
            "Model": name, "Estimator": mm["estimator"],
            "Key hyperparameters": ", ".join(f"{p}={mm['params'].get(p)}" for p in KEY_PARAMS[name]),
            "Score for AUC": mm["score_source"], "Fit time (s)": mm["fit_seconds"],
            "File": mm["file"], "Size (MB)": mm["file_size_mb"],
        })
    st.dataframe(pd.DataFrame(model_rows), width="stretch", hide_index=True)
    st.caption("Library versions used for training: " +
               ", ".join(f"{k} {v}" for k, v in metadata["versions"].items()))

st.html(f'<div class="td-foot">Inference only: the five pipelines are loaded from <code>model/*.pkl</code> and '
        f'never refitted on uploaded data · BITS Pilani M.Tech AIML · Student {STUDENT_ID}</div>')
