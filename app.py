"""
Churn Prediction Streamlit App
Refactored & enhanced from the original notebook/app.py.
- Caching for model/explainer loading
- Safer SHAP plotting (works both locally & on Streamlit Cloud)
- Gauge chart rewritten with pure matplotlib objects (no base64 juggling in main flow)
- Cleaner UI (sidebar inputs, result tabs)
- Input validation & defaults
- Feature engineering kept consistent with training columns
"""

from __future__ import annotations

import io
import base64
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle, Circle

# ---------------------------------------------------------------------
# Config & utils
# ---------------------------------------------------------------------
st.set_page_config(page_title="Churn Prediction", page_icon="📉", layout="wide")

MODEL_PATH = Path("models/model_rfc.pkl")

# Training columns (order matters!)
COLUMNS: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
    "MonthlyCharges",
    "TotalCharges",
    "InternetService_Fiber optic",
    "InternetService_No",
    "Contract_One year",
    "Contract_Two year",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
]

# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_explainer():
    import pickle
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH!s}")
        st.stop()
    model = pickle.load(open(MODEL_PATH, "rb"))
    explainer = shap.TreeExplainer(model)
    return model, explainer

@st.cache_resource(show_spinner=False)

def load_mlp_and_explainer(mlp_path: str = "models/mlp_model.pkl"):
    """Load MLP model safely. If unpickling fails (e.g. numpy MT19937 issue),
    return (None, None) so the UI keeps working with RFC only.
    """
    import pickle, joblib
    p = Path(mlp_path)
    if not p.exists():
        return None, None
    mlp = None
    # try joblib first (more robust across numpy versions)
    try:
        mlp = joblib.load(p)
    except Exception:
        try:
            mlp = pickle.load(open(p, "rb"))
        except Exception as e:
            st.warning(f"MLP model could not be loaded ({e}). It will be ignored.")
            return None, None
    # Build explainer
    try:
        expl = shap.Explainer(mlp)
    except Exception:
        try:
            expl = shap.KernelExplainer(mlp.predict_proba, np.zeros((1, len(COLUMNS))))
        except Exception as e:
            st.warning(f"Failed to create SHAP explainer for MLP ({e}). MLP disabled.")
            return None, None
    return mlp, expl

model, explainer = load_model_and_explainer()
mlp_model, explainer_mlp = load_mlp_and_explainer()

# Small helper to pick which model to use in each tab
def choose_model_ui(label: str):
    """Return (model, explainer, tag). Always offer both RFC and MLP; if MLP failed to load, show an error when chosen."""
    opts = ["RFC", "MLP"]
    choice = st.selectbox(label, opts, key=label)
    if choice == "MLP":
        if mlp_model is None or explainer_mlp is None:
            st.error("MLP model not available (pickle error). Re-export it or fix the path.")
            return model, explainer, "RFC"
        return mlp_model, explainer_mlp, "MLP"
    return model, explainer, "RFC"

# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------
def degree_range(n: int):
    start = np.linspace(0, 180, n + 1, endpoint=True)[:-1]
    end = np.linspace(0, 180, n + 1, endpoint=True)[1:]
    mid_points = start + ((end - start) / 2.0)
    return np.c_[start, end], mid_points


def rot_text(angle: float) -> float:
    """Return rotation for text so labels are upright-ish."""
    return np.degrees(np.radians(angle) * np.pi / np.pi - np.radians(90))


def gauge(
    probability: float,
    labels: List[str] | None = None,
    colors: List[str] | None = None,
) -> bytes:
    """Build a semi-circular gauge and return PNG bytes."""
    if labels is None:
        labels = ["LOW", "MEDIUM", "HIGH", "EXTREME"]
    if colors is None:
        colors = ["#007A00", "#0063BF", "#FFCC00", "#ED1C24"]

    colors = colors[::-1]
    labels = labels[::-1]

    fig, ax = plt.subplots(figsize=(6, 3))
    ang_range, mid_points = degree_range(len(labels))

    for ang, c in zip(ang_range, colors):
        ax.add_patch(Wedge((0.0, 0.0), 0.4, *ang, facecolor="w", lw=2))
        ax.add_patch(Wedge((0.0, 0.0), 0.4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.6))

    for mid, lab in zip(mid_points, labels):
        ax.text(
            0.35 * np.cos(np.radians(mid)),
            0.35 * np.sin(np.radians(mid)),
            lab,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            rotation=rot_text(mid),
        )

    ax.add_patch(Rectangle((-0.4, -0.1), 0.8, 0.1, facecolor="w", lw=2))
    ax.text(0, -0.05, f"Churn Probability {probability:.2f}", ha="center", va="center", fontsize=16, fontweight="bold")

    pos = (1 - probability) * 180
    ax.arrow(
        0,
        0,
        0.225 * np.cos(np.radians(pos)),
        0.225 * np.sin(np.radians(pos)),
        width=0.02,
        head_width=0.06,
        head_length=0.07,
        fc="k",
        ec="k",
    )

    ax.add_patch(Circle((0, 0), radius=0.02, facecolor="k"))
    ax.add_patch(Circle((0, 0), radius=0.01, facecolor="w", zorder=11))

    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("equal")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------
def build_feature_vector() -> pd.DataFrame:
    st.sidebar.header("Input Parameters")

    gender = st.sidebar.selectbox("Gender", [0, 1], index=0, help="0 = Female, 1 = Male (as encoded in training)")
    senior = st.sidebar.checkbox("Senior Citizen")
    partner = st.sidebar.checkbox("Partner")
    dependents = st.sidebar.checkbox("Dependents")
    tenure = st.sidebar.number_input("Tenure (months)", min_value=0, value=1)
    phone_service = st.sidebar.checkbox("Phone Service")
    multiple_lines = st.sidebar.checkbox("Multiple Lines", disabled=not phone_service)

    internet_service = st.sidebar.selectbox("Internet Service", [0, 1, 2], format_func=lambda x: ["No", "DSL", "Fiber optic"][x])
    online_security = st.sidebar.checkbox("Online Security", disabled=internet_service == 0)
    online_backup = st.sidebar.checkbox("Online Backup", disabled=internet_service == 0)
    device_protection = st.sidebar.checkbox("Device Protection", disabled=internet_service == 0)
    tech_support = st.sidebar.checkbox("Tech Support", disabled=internet_service == 0)
    streaming_tv = st.sidebar.checkbox("Streaming TV", disabled=internet_service == 0)
    streaming_movies = st.sidebar.checkbox("Streaming Movies", disabled=internet_service == 0)

    contract = st.sidebar.selectbox("Contract", [0, 1, 2], format_func=lambda x: ["Month-to-month", "One year", "Two year"][x])
    payment_method = st.sidebar.selectbox(
        "Payment Method",
        [0, 1, 2, 3],
        format_func=lambda x: ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"][x],
    )

    monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, value=50.0, step=1.0)
    total_charges = monthly_charges * tenure

    features = [
        gender,
        int(senior),
        int(partner),
        int(dependents),
        tenure,
        int(phone_service),
        int(multiple_lines),
        int(online_security),
        int(online_backup),
        int(device_protection),
        int(tech_support),
        int(streaming_tv),
        int(streaming_movies),
        int(st.sidebar.checkbox("Paperless Billing")),
        monthly_charges,
        total_charges,
        int(internet_service == 2),
        int(internet_service == 0),
        int(contract == 1),
        int(contract == 2),
        int(payment_method == 1),
        int(payment_method == 2),
        int(payment_method == 3),
    ]

    df = pd.DataFrame([features], columns=COLUMNS)
    return df


def plot_shap_force_html(explainer, shap_values_row, data_row: pd.DataFrame) -> str:
    """Return a self-contained HTML string for a single-sample SHAP force plot.
    We always prepend SHAP's JS to avoid the 'Javascript library not loaded' warning.
    """
    exp_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    fp = shap.force_plot(exp_val, shap_values_row, data_row, matplotlib=False)
    # Try to grab HTML payload
    try:
        body = fp.html()
    except AttributeError:
        body = fp.data  # older versions
    return shap.getjs() + body


# ---------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------
input_df = build_feature_vector()

st.title("📉 Churn Prediction App")

tab_pred, tab_shap, tab_lime, tab_shapash, tab_inputs = st.tabs([
    "Prediction", "SHAP", "LIME", "Shapash", "Inputs"
])

# ---------------- Prediction tab ----------------
with tab_pred:
    mdl_pred, expl_pred, tag_pred = choose_model_ui("Model for prediction")
    proba = float(mdl_pred.predict_proba(input_df)[0, 1])

    raw_sv = expl_pred.shap_values(input_df)
    if isinstance(raw_sv, list):
        sv_pred = raw_sv[1]
    else:
        sv_pred = raw_sv

    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric(f"{tag_pred} churn probability", f"{proba:.2%}")
        st.image(gauge(proba), caption="Churn Probability Gauge")
    with col2:
        top_n = 10
        row = np.array(sv_pred[0])
        if row.ndim == 2:
            row = row[:, 1] if row.shape[1] > 1 else row[:, 0]
        shap_abs = pd.Series(np.abs(row), index=COLUMNS).sort_values(ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(6, 4))
        shap_abs[::-1].plot(kind="barh", ax=ax)
        ax.set_xlabel("|SHAP value|")
        ax.set_title(f"Top {top_n} feature impacts ({tag_pred})")
        st.pyplot(fig)

# ---------------- SHAP tab ----------------
with tab_shap:
    mdl_s, expl_s, tag_s = choose_model_ui("Model for SHAP")
    raw_sv = expl_s.shap_values(input_df)
    if isinstance(raw_sv, list):
        sv = raw_sv[1]
    else:
        sv = raw_sv

    st.subheader(f"Force plot (JS) – {tag_s}")
    import streamlit.components.v1 as components
    row = sv[0]
    if getattr(row, "ndim", 1) == 2:
        row = row[:, 1] if row.shape[1] > 1 else row[:, 0]
    html = plot_shap_force_html(expl_s, row, input_df)
    components.html(html, height=320, scrolling=True)

    st.subheader(f"Waterfall plot ({tag_s})")
    try:
        base_val = expl_s.expected_value[1] if isinstance(expl_s.expected_value, (list, np.ndarray)) else expl_s.expected_value
        shap_exp = shap.Explanation(values=row, base_values=base_val, data=input_df.iloc[0], feature_names=COLUMNS)
        shap.plots.waterfall(shap_exp, show=False)
        st.pyplot(bbox_inches="tight")
    except Exception as e:
        st.warning(f"{tag_s} waterfall failed: {e}")

# ---------------- LIME tab ----------------
with tab_lime:
    mdl_l, expl_l, tag_l = choose_model_ui("Model for LIME")
    st.subheader(f"LIME local explanation – {tag_l}")
    try:
        from lime.lime_tabular import LimeTabularExplainer

        @st.cache_resource(show_spinner=False)
        def get_lime_explainer(train_X: pd.DataFrame):
            return LimeTabularExplainer(
                train_X.values,
                feature_names=train_X.columns.tolist(),
                class_names=["No churn", "Churn"],
                discretize_continuous=True,
            )
        # TODO: replace dummy_train with your real training dataset
        dummy_train = pd.DataFrame(np.random.rand(500, len(COLUMNS)), columns=COLUMNS)
        lime_exp = get_lime_explainer(dummy_train)
        exp = lime_exp.explain_instance(input_df.iloc[0].values, mdl_l.predict_proba, num_features=10)
        st.write(exp.as_list())
        fig_lime = exp.as_pyplot_figure()
        st.pyplot(fig_lime)
    except ModuleNotFoundError:
        st.warning("Install LIME: pip install lime")
    except Exception as e:
        st.error(f"LIME error: {e}")

# ---------------- Shapash tab ----------------
with tab_shapash:
    mdl_sh, expl_sh, tag_sh = choose_model_ui("Model for Shapash")
    st.subheader(f"Shapash overview – {tag_sh}")
    try:
        import sys, importlib
        for m in list(sys.modules):
            if m.startswith("shapash"):
                del sys.modules[m]
        from shapash.explainer.smart_explainer import SmartExplainer
        # TODO: replace dummy data with your real X_train, y_train
        dummy_train = pd.DataFrame(np.random.rand(200, len(COLUMNS)), columns=COLUMNS)
        dummy_y = np.random.randint(0, 2, size=len(dummy_train))
        xpl = SmartExplainer(model=mdl_sh)
        xpl.compile(x=dummy_train, y=dummy_y, features_dict={c: c for c in COLUMNS})
        fi = xpl.features_importance().head(10)
        st.write(fi)
    except ModuleNotFoundError:
        st.warning("Install shapash: pip install shapash>=2.3.0")
    except Exception as e:
        st.error("Shapash couldn't run: %s" % str(e))

# ---------------- Inputs tab ----------------
with tab_inputs:
    st.dataframe(input_df)

st.caption("Models: RFC or MLP selectable per tab. Includes SHAP force/waterfall, LIME, and Shapash extracts.")
