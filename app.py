"""
WTI Crude Oil 10-Day Forecast — Streamlit UI
Ensemble: LSTM + CNN + LightGBM + Ridge meta-model
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
import os
import urllib.request
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="WTI Crude Oil Forecast",
    page_icon="🛢️",
    layout="wide",
)

# ── Dark theme styling ────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  .stApp { background-color: #0b0e13; }

  .block-container { padding-top: 2rem; padding-bottom: 2rem; }

  .metric-card {
    background: #131820;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
  }
  .metric-label {
    font-size: 11px;
    color: #6b7a90;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'Space Mono', monospace;
    margin-bottom: 6px;
  }
  .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #e8edf5;
    line-height: 1;
  }
  .metric-value.up { color: #10b981; }
  .metric-value.down { color: #ef4444; }
  .metric-sub { font-size: 11px; color: #6b7a90; margin-top: 4px; }

  .section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: #e8edf5;
    margin-bottom: 0.25rem;
  }
  .section-sub { font-size: 12px; color: #6b7a90; margin-bottom: 1rem; }

  .model-card {
    background: #131820;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
  }
  .model-name {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    margin-bottom: 10px;
    text-transform: uppercase;
  }
  .model-metric-row {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    font-size: 12px;
  }
  .model-metric-row:last-child { border-bottom: none; }
  .model-metric-key { color: #6b7a90; font-family: 'Space Mono', monospace; font-size: 11px; }
  .model-metric-val { font-family: 'Space Mono', monospace; color: #e8edf5; }

  .logo-row {
    display: flex; align-items: center; gap: 12px; margin-bottom: 2rem;
  }
  .logo-mark {
    width: 40px; height: 40px;
    border: 1.5px solid #f59e0b;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    color: #f59e0b;
  }
  .status-live { color: #10b981; font-family: 'Space Mono', monospace; font-size: 12px; }
  .status-error { color: #ef4444; font-family: 'Space Mono', monospace; font-size: 12px; }
  .footer {
    text-align: center;
    padding: 2rem 0 1rem;
    font-size: 11px;
    color: #6b7a90;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.05em;
  }

  div[data-testid="stDataFrame"] { background: #131820; }
  .stButton > button {
    background: transparent;
    border: 1px solid rgba(255,255,255,0.13);
    color: #e8edf5;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    border-radius: 6px;
    padding: 0.4rem 1rem;
  }
  .stButton > button:hover {
    border-color: #f59e0b;
    color: #f59e0b;
    background: #1a2130;
  }
</style>
""", unsafe_allow_html=True)

# ── Model path ────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# ── TensorFlow ────────────────────────────────────────────────
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# ── Data fetchers ─────────────────────────────────────────────
def _fetch_yahoo(n):
    end   = int(datetime.utcnow().timestamp())
    start = int((datetime.utcnow() - timedelta(days=n * 2)).timestamp())
    for host in ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]:
        url = f"https://{host}/v8/finance/chart/CL=F?interval=1d&period1={start}&period2={end}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            result = data["chart"]["result"][0]
            quote  = result["indicators"]["quote"][0]
            closes, opens, highs, lows = quote["close"], quote["open"], quote["high"], quote["low"]
            rows = []
            for i in range(len(closes)):
                if closes[i] is not None:
                    rows.append({"Price": closes[i], "Open": opens[i] or closes[i],
                                 "High": highs[i] or closes[i], "Low": lows[i] or closes[i]})
            if len(rows) >= 60:
                return rows[-n:]
        except Exception:
            continue
    raise RuntimeError("Yahoo Finance unavailable")

def _fetch_stooq(n):
    url = "https://stooq.com/q/d/l/?s=cl.f&i=d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        lines = resp.read().decode().strip().splitlines()
    rows = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            rows.append({"Price": float(parts[4]), "Open": float(parts[1]),
                         "High": float(parts[2]), "Low": float(parts[3])})
        except ValueError:
            continue
    if len(rows) < 60:
        raise RuntimeError("Stooq returned insufficient data")
    return rows[-n:]

def _fetch_alpha_vantage(n):
    url = "https://www.alphavantage.co/query?function=WTI&interval=daily&apikey=demo"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    entries = data.get("data", [])
    rows = []
    for e in sorted(entries, key=lambda x: x["date"]):
        try:
            p = float(e["value"])
            rows.append({"Price": p, "Open": p, "High": p, "Low": p})
        except (ValueError, KeyError):
            continue
    if len(rows) < 60:
        raise RuntimeError("Alpha Vantage returned insufficient data")
    return rows[-n:]

def _fallback_npy(n):
    arts      = load_artifacts()
    scaler    = arts["scaler"]
    cfg       = arts["config"]
    price_idx = cfg["price_index"]
    seq       = np.load(os.path.join(MODELS_DIR, "last_sequence.npy"))
    real      = scaler.inverse_transform(seq)
    prices    = real[:, price_idx].tolist()
    return [{"Price": p, "Open": p, "High": p, "Low": p} for p in prices]

def fetch_live_wti_prices(n=120):
    for name, fn in [("Yahoo Finance", _fetch_yahoo), ("Stooq", _fetch_stooq), ("Alpha Vantage", _fetch_alpha_vantage)]:
        try:
            return fn(n), name
        except Exception:
            continue
    try:
        return _fallback_npy(n), "Saved training sequence"
    except Exception as e:
        raise RuntimeError(f"All data sources failed: {e}")

# ── Feature builder ───────────────────────────────────────────
def build_features(rows):
    df = pd.DataFrame(rows)
    df["Change %"]     = df["Price"].pct_change()
    df["ma_5"]         = df["Price"].rolling(5).mean()
    df["ma_20"]        = df["Price"].rolling(20).mean()
    df["momentum"]     = df["Price"].pct_change(periods=5)
    df["volatility"]   = df["Price"].rolling(10).std()
    df["price_change"] = df["Price"].pct_change()
    delta = df["Price"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))
    df = df.bfill().ffill().fillna(0)
    feature_order = ['Price','Open','High','Low','Change %','ma_5','ma_20','RSI','momentum','volatility','price_change']
    return df[feature_order].values

# ── Model loader ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_artifacts():
    cache = {}
    with open(os.path.join(MODELS_DIR, "config.json")) as f:
        cache["config"] = json.load(f)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        cache["scaler"] = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "lgb_model.pkl"), "rb") as f:
        cache["lgb"] = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "ridge_meta_model.pkl"), "rb") as f:
        cache["ridge"] = pickle.load(f)
    if TF_AVAILABLE:
        try:
            cache["lstm"] = tf.keras.models.load_model(os.path.join(MODELS_DIR, "lstm_model.h5"), compile=False)
            cache["cnn"]  = tf.keras.models.load_model(os.path.join(MODELS_DIR, "cnn_model.h5"),  compile=False)
        except Exception as e:
            st.warning(f"TF models failed: {e}")
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            cache["metrics"] = json.load(f)
    return cache

# ── Forecast engine ───────────────────────────────────────────
def run_forecast(rows):
    arts       = load_artifacts()
    cfg        = arts["config"]
    scaler     = arts["scaler"]
    seq_len    = cfg["seq_len"]
    horizon    = cfg["forecast_horizon"]
    price_idx  = cfg["price_index"]
    n_features = len(cfg["total_fet"])

    features   = build_features(rows)
    features   = features[-seq_len:]
    last_price = float(features[-1, price_idx])
    scaled     = scaler.transform(features)
    x_input    = scaled[np.newaxis, :, :]
    x_flat     = x_input.reshape(1, -1)

    if TF_AVAILABLE and "lstm" in arts and "cnn" in arts:
        lstm_pred = arts["lstm"].predict(x_input, verbose=0).reshape(1, horizon, n_features)
        cnn_pred  = arts["cnn"].predict(x_input, verbose=0).reshape(1, horizon, n_features)
    else:
        returns  = np.diff(features[:, price_idx]) / features[:-1, price_idx]
        avg_ret  = float(np.mean(returns[-10:]))
        fallback = np.tile(features[-1:], (horizon, 1)).copy()
        p = last_price
        for t in range(horizon):
            p = p * (1 + avg_ret * (0.9 ** t))
            fallback[t, price_idx] = p
        fallback_scaled = scaler.transform(fallback)[np.newaxis]
        lstm_pred = fallback_scaled
        cnn_pred  = fallback_scaled

    lgb_pred = arts["lgb"].predict(x_flat).reshape(1, horizon, n_features)
    X_meta   = np.concatenate([lstm_pred.reshape(1, -1), cnn_pred.reshape(1, -1), lgb_pred.reshape(1, -1)], axis=1)
    ensemble_scaled = arts["ridge"].predict(X_meta).reshape(horizon, n_features)
    ensemble_orig   = scaler.inverse_transform(ensemble_scaled)
    forecast_prices = ensemble_orig[:, price_idx].tolist()

    clamped, prev = [], last_price
    for p in forecast_prices:
        if abs(p - last_price) / last_price > 0.25:
            p = prev * (1 + np.sign(p - prev) * 0.01)
        clamped.append(round(p, 2))
        prev = p

    dates, d = [], datetime.today() + timedelta(days=1)
    while len(dates) < horizon:
        if d.weekday() < 5:
            dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)

    return {
        "dates": dates, "prices": clamped,
        "last_actual": round(last_price, 2),
        "model": "LSTM + CNN + LightGBM + Ridge Ensemble" if (TF_AVAILABLE and "lstm" in arts) else "LightGBM + Ridge Ensemble",
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

# ── UI ────────────────────────────────────────────────────────
st.markdown("""
<div class="logo-row">
  <div class="logo-mark">WTI</div>
  <div>
    <div style="font-family:'Space Mono',monospace;font-size:1.1rem;font-weight:700;color:#e8edf5;">Crude Oil Forecast</div>
    <div style="font-size:0.7rem;color:#6b7a90;letter-spacing:0.12em;text-transform:uppercase;">Ensemble ML · 10-Day Outlook</div>
  </div>
</div>
""", unsafe_allow_html=True)

col_status, col_btn = st.columns([6, 1])

with col_btn:
    refresh = st.button("↻ Refresh")

# ── Load data ─────────────────────────────────────────────────
if "forecast" not in st.session_state or refresh:
    with st.spinner("Fetching live prices and running forecast…"):
        try:
            rows, source = fetch_live_wti_prices(120)
            forecast     = run_forecast(rows)
            forecast["source"] = source
            st.session_state["forecast"] = forecast
            st.session_state["rows"]     = rows
            st.session_state["error"]    = None
        except Exception as e:
            st.session_state["error"]    = str(e)
            st.session_state["forecast"] = None

forecast = st.session_state.get("forecast")
rows     = st.session_state.get("rows", [])
error    = st.session_state.get("error")

with col_status:
    if forecast:
        st.markdown(f'<div class="status-live">● live · {datetime.now().strftime("%H:%M:%S")} · Data: {forecast.get("source","")}</div>', unsafe_allow_html=True)
    elif error:
        st.markdown(f'<div class="status-error">✕ error — {error}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#6b7a90;font-family:Space Mono,monospace;font-size:12px;">connecting…</div>', unsafe_allow_html=True)

if error:
    st.error(f"⚠ {error}")

if forecast:
    last   = forecast["last_actual"]
    prices = forecast["prices"]
    dates  = forecast["dates"]
    d1     = prices[0]
    low_p  = min(prices)
    high_p = max(prices)
    chg    = (prices[-1] - last) / last * 100
    chg_dir = "up" if chg >= 0 else "down"
    d1_dir  = "up" if d1 >= last else "down"

    # ── Summary cards ──────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Last Price</div>
            <div class="metric-value">${last:.2f}</div>
            <div class="metric-sub">$/barrel (live)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Day 1 Forecast</div>
            <div class="metric-value {d1_dir}">${d1:.2f}</div>
            <div class="metric-sub">{dates[0]}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">10-Day Range</div>
            <div class="metric-value">${low_p:.2f} – ${high_p:.2f}</div>
            <div class="metric-sub">Low → High</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">10-Day Change</div>
            <div class="metric-value {chg_dir}">{'+'if chg>=0 else ''}{chg:.2f}%</div>
            <div class="metric-sub">vs last actual</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Forecast chart ──────────────────────────────────────
    st.markdown(f"""<div class="section-title">10-DAY PRICE FORECAST</div>
    <div class="section-sub">{forecast["model"]}</div>""", unsafe_allow_html=True)

    # get last 10 actual prices for comparison
    actual_prices = [row["Price"] for row in rows[-10:]]
    actual_dates  = []
    d_act = datetime.today() - timedelta(days=14)
    while len(actual_dates) < 10:
        if d_act.weekday() < 5:
            actual_dates.append(d_act.strftime("%a %b %-d"))
        d_act += timedelta(days=1)

    forecast_labels = [datetime.strptime(d, "%Y-%m-%d").strftime("%a %b %-d") for d in dates]

    # combine for tight y-axis range
    all_prices = actual_prices + prices
    y_min = min(all_prices) * 0.995
    y_max = max(all_prices) * 1.005

    fig = go.Figure()

    # actual prices line
    fig.add_trace(go.Scatter(
        x=actual_dates, y=actual_prices, name="Actual",
        line=dict(color="#10b981", width=2, shape="spline", smoothing=1.3),
        mode="lines+markers",
        marker=dict(color="#10b981", size=5, line=dict(color="#0b0e13", width=1)),
    ))

    # connecting bridge from last actual to first forecast
    fig.add_trace(go.Scatter(
        x=[actual_dates[-1], forecast_labels[0]],
        y=[actual_prices[-1], prices[0]],
        name="Bridge",
        line=dict(color="#f59e0b", width=1.5, dash="dot"),
        mode="lines", showlegend=False,
    ))

    # forecast line
    fig.add_trace(go.Scatter(
        x=forecast_labels, y=prices, name="Forecast",
        line=dict(color="#f59e0b", width=2.5, shape="spline", smoothing=1.3),
        fill="tonexty" if False else None,
        mode="lines+markers",
        marker=dict(color="#f59e0b", size=6, line=dict(color="#0b0e13", width=2)),
    ))

    # shaded forecast area
    fig.add_trace(go.Scatter(
        x=forecast_labels + forecast_labels[::-1],
        y=[p * 1.003 for p in prices] + [p * 0.997 for p in prices[::-1]],
        fill="toself", fillcolor="rgba(245,158,11,0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))

    fig.update_layout(
        paper_bgcolor="#131820", plot_bgcolor="#131820",
        font=dict(family="Space Mono", color="#6b7a90", size=10),
        height=350, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e8edf5")),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.07)"),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.07)",
            tickprefix="$", range=[y_min, y_max],
        ),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Forecast table ──────────────────────────────────────
    st.markdown("""<div class="section-title">DAILY BREAKDOWN</div>
    <div class="section-sub">Next 10 trading days</div>""", unsafe_allow_html=True)

    day_labels = ["1st","2nd","3rd","4th","5th","6th","7th","8th","9th","10th"]
    rows_data  = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        prev  = last if i == 0 else prices[i-1]
        chg_d = price - prev
        chg_p = chg_d / prev * 100
        arrow = "▲" if chg_d >= 0 else "▼"
        rows_data.append({
            "Day":            day_labels[i],
            "Date":           date,
            "Forecast Price": f"${price:.2f}",
            "Change ($)":     f"{arrow} {abs(chg_d):.2f}",
            "Change (%)":     f"{'+'if chg_p>=0 else ''}{chg_p:.2f}%",
        })

    df_table = pd.DataFrame(rows_data)
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    # ── Model performance ───────────────────────────────────
    arts = load_artifacts()
    metrics = arts.get("metrics", {})
    if metrics:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div class="section-title">MODEL PERFORMANCE</div>
        <div class="section-sub">Test set evaluation metrics</div>""", unsafe_allow_html=True)

        colors = {"LSTM": "#f59e0b", "CNN": "#10b981", "LightGBM": "#60a5fa", "Ensemble": "#a78bfa"}
        order  = ["LSTM", "CNN", "LightGBM", "Ensemble"]

        mc1, mc2, mc3, mc4 = st.columns(4)
        for col, name in zip([mc1, mc2, mc3, mc4], order):
            if name not in metrics:
                continue
            m = metrics[name]
            with col:
                st.markdown(f"""<div class="model-card">
                    <div class="model-name" style="color:{colors[name]}">{name}</div>
                    <div class="model-metric-row"><span class="model-metric-key">MAE</span><span class="model-metric-val">${m['MAE']:.4f}</span></div>
                    <div class="model-metric-row"><span class="model-metric-key">RMSE</span><span class="model-metric-val">${m['RMSE']:.4f}</span></div>
                    <div class="model-metric-row"><span class="model-metric-key">R²</span><span class="model-metric-val">{m['R2']:.4f}</span></div>
                    <div class="model-metric-row"><span class="model-metric-key">MAPE</span><span class="model-metric-val">{m['MAPE']:.2f}%</span></div>
                </div>""", unsafe_allow_html=True)

        # bar chart
        maes  = [metrics[n]["MAE"]  for n in order if n in metrics]
        mapes = [metrics[n]["MAPE"] for n in order if n in metrics]
        names = [n for n in order if n in metrics]
        clrs  = [colors[n] for n in names]

        def hex_to_rgba(hex_color, alpha):
            hex_color = hex_color.lstrip("#")
            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="MAE ($)", x=names, y=maes,
            marker_color=[hex_to_rgba(c, 0.6) for c in clrs],
            marker_line_color=clrs, marker_line_width=1.5))
        fig2.add_trace(go.Bar(name="MAPE (%)", x=names, y=mapes,
            marker_color=[hex_to_rgba(c, 0.2) for c in clrs],
            marker_line_color=clrs, marker_line_width=1.5))
        fig2.update_layout(
            paper_bgcolor="#131820", plot_bgcolor="#131820",
            font=dict(family="Space Mono", color="#6b7a90", size=10),
            height=240, margin=dict(l=10, r=10, t=10, b=10),
            barmode="group",
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e8edf5")),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        )
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("""<div class="footer">
  WTI CRUDE OIL FORECAST · LSTM + CNN + LIGHTGBM + RIDGE ENSEMBLE · DATA VIA YAHOO FINANCE
</div>""", unsafe_allow_html=True)
