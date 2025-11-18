# ============================
# streamlit_app.py — Unified FL Dashboard (no dataset selector)
# ============================

import streamlit as st
import pandas as pd
import numpy as np
import io, requests
import altair as alt
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
SERVER_URL = "http://192.168.1.6:5000"   # <- no trailing slash
DATASET_TYPES = ["color","segmented"]  # server-side datasets we aggregate
PAGE_TITLE = "🌿 Federated Plant Disease Detection —  Dashboard"

st.set_page_config(page_title="🌾 Federated Learning Dashboard", page_icon="🌱", layout="wide")

# -----------------------------
# HEADER
# -----------------------------
col_head_l, col_head_c, col_head_r = st.columns([3, 1.2, 1.2])
with col_head_l:
    st.title(PAGE_TITLE)
    st.caption("Monitor training across all datasets !!!.....")

with col_head_c:
    # server status
    try:
        status_resp = requests.get(f"{SERVER_URL}/metrics", timeout=3)
        online = status_resp.status_code == 200
    except Exception:
        online = False

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:8px;margin-top:14px">
          <div style="width:12px;height:12px;border-radius:50%;
               background:{'#2ecc71' if online else '#e74c3c'};"></div>
          <span style="font-weight:600;">Server: {"Online" if online else "Offline"}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_head_r:
    st.write("⏱ Last check")
    st.caption(datetime.now().strftime("%H:%M:%S · %d %b %Y"))

st.markdown("---")

# -----------------------------
# HELPERS
# -----------------------------
@st.cache_data(ttl=20)
def fetch_metrics_all():
    """Return server /metrics JSON or None."""
    try:
        r = requests.get(f"{SERVER_URL}/metrics", timeout=6)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

@st.cache_data(ttl=60)
def fetch_classes(ds: str):
    """Return class list for a dataset (order must match server training)."""
    try:
        r = requests.get(f"{SERVER_URL}/classes", params={"dataset": ds}, timeout=6)
        if r.status_code == 200:
            return r.json().get("classes", [])
    except Exception:
        pass
    return []

def clean_label(name: str) -> str:
    """Make folder-style labels human-readable."""
    if not name:
        return name
    # turn "Tomato___Early_blight" -> "Tomato — Early blight"
    name = name.replace("___", " — ")
    name = name.replace("__", " – ")
    name = name.replace("_", " ")
    # collapse extra spaces
    return " ".join(name.split())

def make_acc_chart(df_long: pd.DataFrame):
    """Altair chart for accuracy over rounds with dataset hue."""
    base = alt.Chart(df_long).encode(
        x=alt.X("round:Q", axis=alt.Axis(title="Round", grid=True)),
        y=alt.Y("accuracy:Q", axis=alt.Axis(title="Accuracy (%)")),
        color=alt.Color("dataset:N", legend=alt.Legend(title="Dataset")),
        tooltip=["dataset:N", "round:Q", alt.Tooltip("accuracy:Q", format=".2f")]
    )
    return base.mark_line(point=True).properties(height=320)

def stat_cards_for_dataset(df: pd.DataFrame, title: str):
    if df is None or df.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{title} · Rounds", "0")
        c2.metric(f"{title} · Best Acc", "-")
        c3.metric(f"{title} · Last Acc", "-")
        return
    best = float(df["accuracy"].max())
    last = float(df["accuracy"].iloc[-1])
    last_r = int(df["round"].iloc[-1])
    delta = last - (float(df["accuracy"].iloc[-2]) if len(df) > 1 else 0.0)
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{title} · Rounds", str(last_r))
    c2.metric(f"{title} · Best Acc", f"{best:.2f}%")
    c3.metric(f"{title} · Last Acc", f"{last:.2f}%", delta=f"{delta:+.2f}%")

# -----------------------------
# METRICS SECTION (one view for all datasets)
# -----------------------------
metrics_json = fetch_metrics_all()

st.subheader("📊 Federated Learning Metrics (All datasets)")

if not metrics_json:
    st.info("No metrics yet (or server offline).")
else:
    # build a long dataframe: dataset | round | accuracy
    frames = []
    for ds in DATASET_TYPES:
        info = metrics_json.get(ds, {})
        rows = info.get("metrics", [])
        if rows:
            df = pd.DataFrame(rows)
            if "round" in df and "accuracy" in df:
                df["dataset"] = ds
                frames.append(df[["dataset", "round", "accuracy"]])
    if not frames:
        st.info("Training not started yet.")
    else:
        df_long = pd.concat(frames, ignore_index=True).sort_values(["dataset", "round"])
        # show cards per dataset
        for ds in DATASET_TYPES:
            stat_cards_for_dataset(df_long[df_long["dataset"] == ds], ds.capitalize())

        st.markdown("#### Accuracy over rounds (combined)")
        st.altair_chart(make_acc_chart(df_long).interactive(), use_container_width=True)

        with st.expander("View raw metrics table"):
            st.dataframe(df_long, use_container_width=True, hide_index=True)

# -----------------------------
# PREDICTION (single unified model call)
# Server decides which dataset model to use; we map to a readable class name.
# -----------------------------
st.markdown("---")
st.header("🧠  Plant Disease Prediction ")

left, right = st.columns([1, 2])

with left:
    uploaded = st.file_uploader("Upload Leaf Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    st.caption("Tip: one clear leaf per image works best.")

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    left.image(image, caption="Leaf Uploaded", use_container_width=True)

    if left.button("🚀 Predict Disease"):
        # serialize and send to server
        buf = io.BytesIO()
        image.save(buf, format="JPEG"); buf.seek(0)
        files = {"file": ("leaf.jpg", buf.getvalue())}

        with st.spinner("Contacting server and running inference..."):
            try:
                r = requests.post(f"{SERVER_URL}/predict", files=files, timeout=25)
            except Exception as e:
                st.error(f"Connection failed: {e}")
                r = None

        if not r or r.status_code != 200:
            st.error("Server error.")
        else:
            res = r.json()
            # expected keys (best effort)
            used_ds = res.get("dataset") or res.get("used_dataset") or "color"
            pred_idx = res.get("predicted_class")
            class_name = res.get("class_name")  # if server already returns string
            conf = float(res.get("confidence", 0.0))

            # Fallback: map index -> class name for the dataset the server used
            if not class_name and isinstance(pred_idx, int):
                ds_classes = fetch_classes(used_ds)
                if 0 <= pred_idx < len(ds_classes):
                    class_name = ds_classes[pred_idx]

            # Final clean-up so you never see "class_24"
            pretty = clean_label(class_name or f"class_{pred_idx}")

            st.success(f"✅ Detected: {pretty}")
            st.caption(f"Model used: **{used_ds.upper()}**")

            # Confidence bar (0..1 expected)
            st.progress(min(max(conf, 0.0), 1.0))
            st.write(f"Confidence: **{conf*100:.2f}%**")

            if conf >= 0.85:
                st.balloons()
else:
    right.info("Upload an image on the left to run a prediction.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
f1, f2 = st.columns(2)
with f1:
    st.caption(f"✅ Server: {SERVER_URL}")
with f2:
    st.caption("SEC PROJECT !!!!")


