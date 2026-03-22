"""
app.py — IoT Dynamic Threat Isolation System
Streamlit UI — Production Grade
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import random
import os
import hashlib
import json

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DTIS · IoT Threat Command",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Master CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ═══════════════════════════════════════════════
   FONTS
═══════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Syne:wght@400;500;700;800&display=swap');

/* ═══════════════════════════════════════════════
   RESET & BASE
═══════════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg-void:       #020608;
  --bg-deep:       #060d12;
  --bg-panel:      #091520;
  --bg-surface:    #0d1e2e;
  --bg-raised:     #112233;
  --border-dim:    rgba(0, 200, 255, 0.08);
  --border-mid:    rgba(0, 200, 255, 0.18);
  --border-bright: rgba(0, 200, 255, 0.45);
  --cyan:          #00c8ff;
  --cyan-dim:      rgba(0, 200, 255, 0.6);
  --cyan-glow:     rgba(0, 200, 255, 0.15);
  --red:           #ff3b5c;
  --red-dim:       rgba(255, 59, 92, 0.12);
  --green:         #00e8a2;
  --green-dim:     rgba(0, 232, 162, 0.1);
  --amber:         #ffb340;
  --amber-dim:     rgba(255, 179, 64, 0.1);
  --text-bright:   #e8f4ff;
  --text-mid:      #7ca8c8;
  --text-dim:      #3a6080;
  --mono:          'IBM Plex Mono', monospace;
  --sans:          'Syne', sans-serif;
}

html, body { scroll-behavior: smooth; }

/* ── App shell ── */
.stApp {
  background-color: var(--bg-void) !important;
  background-image:
    radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0,200,255,0.06) 0%, transparent 60%),
    repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(0,200,255,0.025) 39px, rgba(0,200,255,0.025) 40px),
    repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(0,200,255,0.015) 39px, rgba(0,200,255,0.015) 40px);
  color: var(--text-bright) !important;
  font-family: var(--mono) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
.block-container {
  max-width: 1400px !important;
  padding: 2rem 2.5rem !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--bg-deep) !important;
  border-right: 1px solid var(--border-mid) !important;
}
[data-testid="stSidebar"] * { font-family: var(--mono) !important; }

/* ── All text elements ── */
p, li, span, label, div {
  font-family: var(--mono) !important;
}

/* Preserve material icon ligatures (e.g., arrow_right) as icons */
span[class*="material-symbols"],
span[class*="material-icons"],
i[class*="material-symbols"],
i[class*="material-icons"] {
  font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons", sans-serif !important;
  letter-spacing: normal !important;
  word-spacing: normal !important;
  text-transform: none !important;
  white-space: nowrap !important;
  display: inline-flex !important;
  align-items: center !important;
  line-height: 1 !important;
  font-variation-settings: "FILL" 0, "wght" 400, "GRAD" 0, "opsz" 20;
}
h1, h2, h3, h4, h5, h6 {
  font-family: var(--sans) !important;
  color: var(--text-bright) !important;
}

/* ═══════════════════════════════════════════════
   CUSTOM COMPONENTS
═══════════════════════════════════════════════ */

/* ── Top nav bar ── */
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem 1.5rem;
  background: var(--bg-deep);
  border: 1px solid var(--border-mid);
  border-radius: 4px;
  margin-bottom: 1.75rem;
  position: relative;
  overflow: hidden;
}
.topbar::before {
  content: '';
  position: absolute; inset: 0;
  background: linear-gradient(90deg, rgba(0,200,255,0.04) 0%, transparent 60%);
  pointer-events: none;
}
.topbar-left { display: flex; align-items: center; gap: 1rem; }
.topbar-logo {
  width: 36px; height: 36px;
  border: 1.5px solid var(--cyan);
  border-radius: 4px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.1rem;
  box-shadow: 0 0 12px rgba(0,200,255,0.3), inset 0 0 8px rgba(0,200,255,0.1);
}
.topbar-title {
  font-family: var(--sans) !important;
  font-size: 1rem;
  font-weight: 800;
  color: var(--text-bright) !important;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.topbar-subtitle {
  font-size: 0.62rem;
  color: var(--text-dim);
  letter-spacing: 0.2em;
  text-transform: uppercase;
}
.topbar-right { display: flex; align-items: center; gap: 1.5rem; }
.status-dot {
  display: flex; align-items: center; gap: 0.4rem;
  font-size: 0.68rem; color: var(--text-mid);
  letter-spacing: 0.1em;
}
.dot {
  width: 6px; height: 6px; border-radius: 50%;
  animation: pulse-dot 2s ease-in-out infinite;
}
.dot-green { background: var(--green); box-shadow: 0 0 6px var(--green); }
.dot-amber { background: var(--amber); box-shadow: 0 0 6px var(--amber); animation-delay: 0.5s; }
@keyframes pulse-dot {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
.sys-time {
  font-size: 0.7rem;
  color: var(--cyan-dim);
  letter-spacing: 0.08em;
  border: 1px solid var(--border-dim);
  padding: 0.25rem 0.6rem;
  border-radius: 2px;
}

/* ── Section label ── */
.sec-label {
  font-size: 0.6rem;
  letter-spacing: 0.25em;
  text-transform: uppercase;
  color: var(--text-dim);
  font-family: var(--mono) !important;
  margin-bottom: 0.6rem;
  display: flex; align-items: center; gap: 0.5rem;
}
.sec-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border-dim);
}

/* ── KPI card ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; margin-bottom: 1.75rem; background: var(--border-dim); border-radius: 6px; overflow: hidden; }
.kpi-card {
  background: var(--bg-panel);
  padding: 1.25rem 1.5rem;
  position: relative;
  transition: background 0.2s;
}
.kpi-card:hover { background: var(--bg-surface); }
.kpi-card::before {
  content: attr(data-accent);
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: var(--accent-color, var(--cyan));
  box-shadow: 0 0 8px var(--accent-color, var(--cyan));
}
.kpi-label {
  font-size: 0.6rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--text-dim);
  margin-bottom: 0.5rem;
}
.kpi-value {
  font-family: var(--sans) !important;
  font-size: 2rem;
  font-weight: 800;
  line-height: 1;
  color: var(--text-bright);
  margin-bottom: 0.3rem;
}
.kpi-sub {
  font-size: 0.6rem;
  color: var(--text-dim);
  letter-spacing: 0.1em;
}

/* ── Panel ── */
.panel {
  background: var(--bg-panel);
  border: 1px solid var(--border-dim);
  border-radius: 6px;
  padding: 1.25rem 1.5rem;
  margin-bottom: 1rem;
  position: relative;
  overflow: hidden;
}
.panel::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border-mid), transparent);
}

/* ── Config row ── */
.config-row {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr 1fr auto;
  gap: 1rem;
  align-items: end;
  background: var(--bg-panel);
  border: 1px solid var(--border-dim);
  border-radius: 6px;
  padding: 1.25rem 1.5rem;
  margin-bottom: 1.75rem;
}

/* ── Streamlit widget overrides ── */
[data-testid="stSelectbox"] > div > div {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-mid) !important;
  border-radius: 3px !important;
  color: var(--text-bright) !important;
  font-family: var(--mono) !important;
  font-size: 0.78rem !important;
}
[data-testid="stSlider"] { padding: 0 !important; }
[data-testid="stSlider"] > div > div > div > div {
  background: var(--cyan) !important;
}
[data-testid="stSlider"] > div > div > div {
  background: var(--border-mid) !important;
}
label[data-testid="stWidgetLabel"] p {
  font-size: 0.62rem !important;
  letter-spacing: 0.18em !important;
  text-transform: uppercase !important;
  color: var(--text-dim) !important;
  font-family: var(--mono) !important;
}

/* ── Primary button ── */
.stButton > button {
  background: transparent !important;
  border: 1px solid var(--cyan) !important;
  color: var(--cyan) !important;
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  padding: 0.6rem 1.4rem !important;
  border-radius: 3px !important;
  transition: all 0.15s ease !important;
  position: relative !important;
  overflow: hidden !important;
}
.stButton > button::before {
  content: '';
  position: absolute; inset: 0;
  background: var(--cyan-glow);
  opacity: 0;
  transition: opacity 0.15s;
}
.stButton > button:hover {
  background: var(--cyan-glow) !important;
  box-shadow: 0 0 20px rgba(0,200,255,0.25), inset 0 0 20px rgba(0,200,255,0.05) !important;
  transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Chart containers ── */
[data-testid="stVegaLiteChart"],
[data-testid="stArrowVegaLiteChart"] {
  background: var(--bg-surface) !important;
  border-radius: 4px !important;
  border: 1px solid var(--border-dim) !important;
}

/* ── DataFrame ── */
[data-testid="stDataFrame"] {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-dim) !important;
  border-radius: 4px !important;
  font-size: 0.72rem !important;
  font-family: var(--mono) !important;
}
[data-testid="stDataFrame"] th {
  background: var(--bg-raised) !important;
  color: var(--text-dim) !important;
  font-size: 0.6rem !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  border-bottom: 1px solid var(--border-mid) !important;
}
[data-testid="stDataFrame"] td {
  color: var(--text-mid) !important;
  border-bottom: 1px solid var(--border-dim) !important;
}

/* ── Metrics (native) ── */
[data-testid="stMetric"] {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-dim) !important;
  border-radius: 4px !important;
  padding: 0.8rem 1rem !important;
}
[data-testid="stMetricLabel"] p {
  font-size: 0.58rem !important;
  letter-spacing: 0.18em !important;
  text-transform: uppercase !important;
  color: var(--text-dim) !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--sans) !important;
  font-size: 1.6rem !important;
  font-weight: 800 !important;
  color: var(--cyan) !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div { background: var(--border-dim) !important; border-radius: 1px !important; }
[data-testid="stProgressBar"] > div > div { background: var(--cyan) !important; box-shadow: 0 0 8px var(--cyan) !important; }

/* ── Alert / info ── */
[data-testid="stAlert"] {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-mid) !important;
  border-radius: 4px !important;
  font-family: var(--mono) !important;
  font-size: 0.78rem !important;
  color: var(--text-mid) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--cyan) !important; }

/* ═══════════════════════════════════════════════
   THREAT LOG ENTRIES
═══════════════════════════════════════════════ */
.log-entry {
  display: grid;
  grid-template-columns: 90px 100px 1fr 90px 110px;
  gap: 0;
  align-items: stretch;
  border-bottom: 1px solid var(--border-dim);
  font-size: 0.7rem;
  font-family: var(--mono);
  transition: background 0.1s;
  min-height: 38px;
}
.log-entry:hover { background: rgba(0,200,255,0.03); }
.log-entry:first-child { border-top: 1px solid var(--border-dim); }

.log-cell {
  padding: 0.6rem 0.8rem;
  display: flex; align-items: center;
  border-right: 1px solid var(--border-dim);
  color: var(--text-mid);
}
.log-cell:last-child { border-right: none; }

.log-idx  { color: var(--text-dim); font-size: 0.62rem; }
.log-time { color: var(--text-dim); font-size: 0.62rem; font-variant-numeric: tabular-nums; }

.badge {
  display: inline-flex; align-items: center; gap: 0.35rem;
  font-size: 0.6rem; font-weight: 600;
  letter-spacing: 0.12em; text-transform: uppercase;
  padding: 0.2rem 0.55rem;
  border-radius: 2px;
}
.badge-normal {
  color: var(--green);
  background: var(--green-dim);
  border: 1px solid rgba(0,232,162,0.2);
}
.badge-attack {
  color: var(--red);
  background: var(--red-dim);
  border: 1px solid rgba(255,59,92,0.2);
}
.badge-isolated {
  color: var(--amber);
  background: var(--amber-dim);
  border: 1px solid rgba(255,179,64,0.2);
}

.conf-bar-wrap {
  display: flex; align-items: center; gap: 0.5rem;
  width: 100%;
}
.conf-bar {
  flex: 1; height: 3px;
  background: var(--border-dim);
  border-radius: 2px;
  overflow: hidden;
}
.conf-bar-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.4s ease;
}
.conf-val {
  font-size: 0.6rem;
  color: var(--text-dim);
  font-variant-numeric: tabular-nums;
  min-width: 34px;
  text-align: right;
}

/* Isolated row glow */
.log-entry-isolated {
  background: linear-gradient(90deg, rgba(255,179,64,0.04) 0%, transparent 100%);
}
.log-entry-attack {
  background: linear-gradient(90deg, rgba(255,59,92,0.04) 0%, transparent 100%);
}

/* ── Log header ── */
.log-header {
  display: grid;
  grid-template-columns: 90px 100px 1fr 90px 110px;
  gap: 0;
  border-bottom: 1px solid var(--border-mid);
  padding: 0.4rem 0;
  font-size: 0.56rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--text-dim);
}
.log-header-cell {
  padding: 0 0.8rem;
  border-right: 1px solid var(--border-dim);
}
.log-header-cell:last-child { border-right: none; }

/* ── Matrix heatmap ── */
.cm-grid { display: grid; grid-template-columns: auto 1fr 1fr; gap: 2px; font-size: 0.68rem; }
.cm-cell {
  display: flex; align-items: center; justify-content: center;
  height: 52px; border-radius: 3px;
  font-weight: 600; font-family: var(--sans) !important;
  font-size: 1.1rem;
}
.cm-label {
  display: flex; align-items: center; justify-content: flex-end;
  padding-right: 0.6rem; color: var(--text-dim);
  font-size: 0.58rem; letter-spacing: 0.1em; text-transform: uppercase;
}
.cm-header { display: flex; align-items: center; justify-content: center; color: var(--text-dim); font-size: 0.58rem; letter-spacing: 0.1em; text-transform: uppercase; height: 28px; }
.cm-tp { background: rgba(0,232,162,0.15); color: var(--green); border: 1px solid rgba(0,232,162,0.2); }
.cm-fp { background: rgba(255,59,92,0.1);  color: var(--red);   border: 1px solid rgba(255,59,92,0.15); }
.cm-fn { background: rgba(255,179,64,0.1); color: var(--amber); border: 1px solid rgba(255,179,64,0.15); }
.cm-tn { background: rgba(0,200,255,0.08); color: var(--cyan);  border: 1px solid rgba(0,200,255,0.12); }

/* ── Scan line animation on hero ── */
@keyframes scanline {
  0%   { transform: translateY(-100%); }
  100% { transform: translateY(100vh); }
}

/* ── Divider ── */
.hdivider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border-mid), transparent);
  margin: 1.5rem 0;
}

/* ── Onboarding cards ── */
.onboard-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background: var(--border-dim); border-radius: 6px; overflow: hidden; margin-top: 1rem; }
.onboard-card {
  background: var(--bg-panel);
  padding: 1.5rem;
  position: relative;
}
.onboard-step {
  font-family: var(--sans) !important;
  font-size: 2.5rem;
  font-weight: 800;
  color: var(--border-mid);
  line-height: 1;
  margin-bottom: 0.75rem;
}
.onboard-title {
  font-family: var(--sans) !important;
  font-size: 0.85rem;
  font-weight: 700;
  color: var(--text-bright);
  margin-bottom: 0.4rem;
  letter-spacing: 0.05em;
}
.onboard-desc {
  font-size: 0.68rem;
  color: var(--text-dim);
  line-height: 1.6;
  letter-spacing: 0.02em;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
  background: transparent !important;
  border: 1px solid var(--border-mid) !important;
  color: var(--text-mid) !important;
  font-family: var(--mono) !important;
  font-size: 0.65rem !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  padding: 0.45rem 0.9rem !important;
  border-radius: 3px !important;
}
[data-testid="stDownloadButton"] > button:hover {
  border-color: var(--cyan) !important;
  color: var(--cyan) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-dim) !important;
  border-radius: 4px !important;
}
[data-testid="stExpander"] summary {
  font-size: 0.7rem !important;
  letter-spacing: 0.1em !important;
  color: var(--text-mid) !important;
  font-family: var(--mono) !important;
}

/* Keep Streamlit icon ligatures rendered as icons, not plain text */
[data-testid="stExpander"] .material-symbols-rounded,
[data-testid="stExpander"] .material-icons,
[data-testid="stExpander"] .material-icons-round {
  font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important;
  letter-spacing: normal !important;
  text-transform: none !important;
  font-variation-settings: "FILL" 0, "wght" 400, "GRAD" 0, "opsz" 20;
}
</style>
""", unsafe_allow_html=True)


# ── Lazy imports ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_imports():
  from data_loader import load_data
  from model import train_model, predict, evaluate_model, get_attack_scores, load_saved_artifacts, save_training_meta, USE_KERAS
  return load_data, train_model, predict, evaluate_model, get_attack_scores, load_saved_artifacts, save_training_meta, USE_KERAS


# ── Session state ─────────────────────────────────────────────────────────────
for key, val in {
    "trained": False, "model": None, "scaler": None,
    "X": None, "y": None, "report": None,
    "history": None, "cm": None, "feature_names": [],
  "inf_results": None, "seq_len": 10,
  "train_device": "", "gen_eval": None,
  "iso_threshold_slider": 0.75,
  "threshold_profiles": {},
  "threshold_profiles_loaded": False,
  "last_profile_device": "",
  "show_architecture": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_UNSW_PATH = os.path.join(PROJECT_ROOT, "data", "UNSW-NB15_1.csv")
DEFAULT_NBAIOT_BENIGN_PATH = os.path.join(PROJECT_ROOT, "data", "Ennio_Doorbell", "benign_traffic.csv")
DEFAULT_NBAIOT_ATTACK_PATH = os.path.join(PROJECT_ROOT, "data", "Ennio_Doorbell", "gafgyt_attacks.rar")
THRESHOLD_PROFILE_PATH = os.path.join(PROJECT_ROOT, "threshold_profiles.json")


def _load_threshold_profiles():
  if not os.path.isfile(THRESHOLD_PROFILE_PATH):
    return {}
  try:
    with open(THRESHOLD_PROFILE_PATH, "r", encoding="utf-8") as f:
      raw = json.load(f)
    if not isinstance(raw, dict):
      return {}
    profiles = {}
    for k, v in raw.items():
      try:
        profiles[str(k)] = float(v)
      except Exception:
        continue
    return profiles
  except Exception:
    return {}


def _save_threshold_profiles(profiles: dict):
  try:
    with open(THRESHOLD_PROFILE_PATH, "w", encoding="utf-8") as f:
      json.dump(profiles, f, indent=2)
  except Exception:
    pass


if not st.session_state["threshold_profiles_loaded"]:
  st.session_state["threshold_profiles"] = _load_threshold_profiles()
  st.session_state["threshold_profiles_loaded"] = True


def _discover_nbaiot_device_paths(project_root: str):
  """
  Build a device -> paths map for N-BaIoT folders.
  Prefers gafgyt attack archive, then mirai, then any *_attacks.rar.
  """
  candidates = []
  data_root = os.path.join(project_root, "data")
  if os.path.isdir(data_root):
    candidates.extend([os.path.join(data_root, d) for d in os.listdir(data_root)])
  candidates.extend([os.path.join(project_root, d) for d in os.listdir(project_root)])

  device_map = {}
  seen = set()
  for folder in candidates:
    if folder in seen or not os.path.isdir(folder):
      continue
    seen.add(folder)

    benign_csv = os.path.join(folder, "benign_traffic.csv")
    if not os.path.isfile(benign_csv):
      continue

    attack_candidates = [
      os.path.join(folder, "gafgyt_attacks.rar"),
      os.path.join(folder, "mirai_attacks.rar"),
    ]

    # Fallback to any *_attacks.rar in this device folder.
    attack_candidates.extend(
      [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith("_attacks.rar")
      ]
    )

    attack_path = next((p for p in attack_candidates if os.path.isfile(p)), "")
    device_map[os.path.basename(folder)] = {
      "benign": benign_csv,
      "attack": attack_path,
    }

  return device_map


def _file_fingerprint(path: str):
    if not path:
        return ""
    try:
        stat = os.stat(path)
        return f"{os.path.abspath(path)}|{int(stat.st_mtime)}|{stat.st_size}"
    except Exception:
        return os.path.abspath(path)


def _build_data_signature(mode: str, kwargs: dict):
    if mode == "demo":
        payload = "demo"
    elif mode == "unsw":
        payload = f"unsw|{_file_fingerprint(kwargs.get('csv_path', ''))}"
    elif mode == "nbaiot":
        benign_fp = _file_fingerprint(kwargs.get("benign_csv", ""))
        attack_path = kwargs.get("attack_path", "")
        if attack_path and os.path.isdir(attack_path):
            parts = []
            try:
                for f in sorted(os.listdir(attack_path)):
                    if f.lower().endswith(".csv"):
                        parts.append(_file_fingerprint(os.path.join(attack_path, f)))
            except Exception:
                parts.append(os.path.abspath(attack_path))
            attack_fp = "|".join(parts) if parts else os.path.abspath(attack_path)
        else:
            attack_fp = _file_fingerprint(attack_path)
        payload = f"nbaiot|{benign_fp}|{attack_fp}"
    else:
        payload = mode
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ── Top navigation bar ────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-left">
    <div class="topbar-logo">⬡</div>
    <div>
      <div class="topbar-title">DTIS &nbsp;·&nbsp; Dynamic Threat Isolation</div>
      <div class="topbar-subtitle">IoT Network Intrusion Detection &amp; Containment System</div>
    </div>
  </div>
  <div class="topbar-right">
    <div class="status-dot"><span class="dot dot-green"></span> SYSTEM NOMINAL</div>
    <div class="status-dot"><span class="dot dot-amber"></span> MONITOR ACTIVE</div>
    <div class="sys-time">v1.0.0 &nbsp;|&nbsp; LSTM ENGINE</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Configuration Row ─────────────────────────────────────────────────────────
st.markdown('<div class="sec-label">System Configuration</div>', unsafe_allow_html=True)

with st.container():
    c1, c2, c3, c4, c5 = st.columns([2, 1.2, 1.2, 1.2, 1])

    with c1:
        data_mode = st.selectbox(
            "Data Source",
            ["demo", "unsw", "nbaiot"],
            format_func=lambda x: {
                "demo":   "⬡  Synthetic Demo Dataset",
                "unsw":   "◈  UNSW-NB15  (CSV Path)",
                "nbaiot": "◈  N-BaIoT  (CSV Path)"
            }[x]
        )

    with c2:
        seq_len = st.slider("Window Size", 5, 30, 10)

    with c3:
        epochs = st.slider("Epochs", 5, 50, 15)

    with c4:
      iso_threshold = st.slider("Isolation Threshold", 0.50, 0.99, 0.75, key="iso_threshold_slider")

    with c5:
        st.write("")
        train_btn = st.button("▶  TRAIN", use_container_width=True)
        force_retrain = st.checkbox("Force retrain", value=False)

# Dataset path inputs (conditional)
if data_mode == "unsw":
    unsw_path = st.text_input("UNSW-NB15 CSV Path", value=DEFAULT_UNSW_PATH, placeholder="data/UNSW-NB15_1.csv")
elif data_mode == "nbaiot":
  nbaiot_devices = _discover_nbaiot_device_paths(PROJECT_ROOT)
  device_options = sorted(nbaiot_devices.keys())
  default_device_name = "Ennio_Doorbell" if "Ennio_Doorbell" in nbaiot_devices else ""
  if not default_device_name and device_options:
    default_device_name = device_options[0]

  if device_options:
    default_idx = device_options.index(default_device_name)
    selected_nbaiot_device = st.selectbox("N-BaIoT Device", device_options, index=default_idx)
    st.session_state["selected_nbaiot_device"] = selected_nbaiot_device

    saved_profiles = st.session_state.get("threshold_profiles", {})
    last_profile_device = st.session_state.get("last_profile_device", "")
    if selected_nbaiot_device != last_profile_device:
      saved_threshold = saved_profiles.get(selected_nbaiot_device)
      if saved_threshold is not None:
        st.session_state["iso_threshold_slider"] = float(saved_threshold)
        st.caption(f"Loaded saved threshold {float(saved_threshold):.3f} for {selected_nbaiot_device}")
      st.session_state["last_profile_device"] = selected_nbaiot_device

    auto_benign = nbaiot_devices[selected_nbaiot_device]["benign"]
    auto_attack = nbaiot_devices[selected_nbaiot_device]["attack"]
    nbaiot_benign = auto_benign
    nbaiot_attack = auto_attack
    if not auto_attack:
      auto_attack = DEFAULT_NBAIOT_ATTACK_PATH
      nbaiot_attack = auto_attack
  else:
    selected_nbaiot_device = ""
    auto_benign = DEFAULT_NBAIOT_BENIGN_PATH
    auto_attack = DEFAULT_NBAIOT_ATTACK_PATH

    col_a, col_b = st.columns(2)
    with col_a:
        nbaiot_benign = st.text_input(
            "Benign CSV",
      value=auto_benign,
            placeholder="data/Ennio_Doorbell/benign_traffic.csv"
        )
    with col_b:
        nbaiot_attack = st.text_input(
            "Attack RAR or extracted folder",
      value=auto_attack,
            placeholder="data/Ennio_Doorbell/gafgyt_attacks.rar"
        )
  if selected_nbaiot_device:
    st.caption(f"Auto-selected device folder: {selected_nbaiot_device}")

st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)


# ── Training logic ─────────────────────────────────────────────────────────────
if train_btn:
    load_data, train_model, predict, evaluate_model, get_attack_scores, load_saved_artifacts, save_training_meta, use_keras_backend = get_imports()

    prog_bar  = st.progress(0, text="Initialising pipeline...")
    status_ph = st.empty()

    steps = ["Parsing dataset...", "Normalising features...", "Building LSTM graph...", "Training..."]
    for i, step in enumerate(steps):
        time.sleep(0.35)
        prog_bar.progress((i + 1) * 20, text=step)

    try:
        kwargs = {}
        if data_mode == "unsw":
            unsw_csv_path = unsw_path.strip() if unsw_path else ""
            if not unsw_csv_path:
                unsw_csv_path = DEFAULT_UNSW_PATH
            if not os.path.isabs(unsw_csv_path):
                unsw_csv_path = os.path.join(PROJECT_ROOT, unsw_csv_path)
            kwargs["csv_path"] = unsw_csv_path
        elif data_mode == "nbaiot":
            benign_csv_path = nbaiot_benign.strip() if nbaiot_benign else ""
            attack_data_path = nbaiot_attack.strip() if nbaiot_attack else ""

            if not benign_csv_path:
                benign_csv_path = auto_benign
            if not benign_csv_path:
                raise ValueError("Please provide a Benign CSV path for N-BaIoT.")

            if not attack_data_path:
                attack_data_path = auto_attack or DEFAULT_NBAIOT_ATTACK_PATH
            if not attack_data_path:
                raise ValueError("Please provide an Attack RAR/CSV/folder path for N-BaIoT.")

            if not os.path.isabs(benign_csv_path):
                benign_csv_path = os.path.join(PROJECT_ROOT, benign_csv_path)
            if not os.path.isabs(attack_data_path):
                attack_data_path = os.path.join(PROJECT_ROOT, attack_data_path)

            if not os.path.exists(benign_csv_path):
                raise FileNotFoundError(f"Benign CSV not found: {benign_csv_path}")
            if not os.path.exists(attack_data_path):
                raise FileNotFoundError(f"Attack data path not found: {attack_data_path}")

            kwargs["benign_csv"] = benign_csv_path
            kwargs["attack_path"] = attack_data_path
        X, y, feat_names = load_data(mode=data_mode, **kwargs)

        expected_meta = {
            "data_signature": _build_data_signature(data_mode, kwargs),
            "seq_len": int(seq_len),
            "n_features": int(X.shape[1]),
            "n_classes": int(len(np.unique(y))),
            "backend": "keras" if use_keras_backend else "rf",
            "eval_protocol": "contiguous_split_v2",
        }
    except Exception as e:
        st.error(f"Data error: {e}")
        st.stop()

    if not force_retrain:
      scaler_saved, model_saved, hist_saved, report_saved, cm_saved, reason = load_saved_artifacts(expected_meta)
      if scaler_saved is not None and model_saved is not None:
        st.info("Loaded compatible pre-trained model. Enable 'Force retrain' to train again.")
        trained_device = selected_nbaiot_device if data_mode == "nbaiot" else ""
        st.session_state.update({
          "trained": True, "model": model_saved, "scaler": scaler_saved,
          "X": X, "y": y, "report": report_saved,
          "history": hist_saved, "cm": cm_saved,
          "feature_names": feat_names, "seq_len": seq_len,
          "inf_results": None, "train_device": trained_device,
        })
        st.rerun()
      else:
        st.caption(f"Retraining required: {reason}")

    try:
        hist, report, cm, scaler, model = train_model(X, y, seq_len=seq_len, epochs=epochs)
    except KeyboardInterrupt:
        st.warning("Training stopped by user.")
        st.stop()
    except Exception as e:
        if "interrupted by user" in str(e).lower():
            st.warning("Training stopped by user.")
        else:
            st.error(f"Training error: {e}")
        st.stop()

    save_training_meta({
        "data_signature": expected_meta["data_signature"],
        "seq_len": expected_meta["seq_len"],
        "n_features": expected_meta["n_features"],
        "n_classes": expected_meta["n_classes"],
        "backend": expected_meta["backend"],
      "eval_protocol": expected_meta["eval_protocol"],
        "history": hist,
        "report": report,
        "cm": cm,
    })

    prog_bar.progress(100, text="Model ready.")
    time.sleep(0.3)
    prog_bar.empty()

    st.session_state.update({
        "trained": True, "model": model, "scaler": scaler,
        "X": X, "y": y, "report": report, "history": hist,
        "cm": cm, "feature_names": feat_names, "seq_len": seq_len,
      "inf_results": None,
      "train_device": selected_nbaiot_device if data_mode == "nbaiot" else "",
      "gen_eval": None,
    })
    st.rerun()


# ── Dashboard (post-training) ─────────────────────────────────────────────────
if st.session_state.trained:
    report  = st.session_state.report
    history = st.session_state.history
    cm      = st.session_state.cm
    seq_len = st.session_state.seq_len

    acc  = report.get("accuracy", 0)
    prec = report.get("1", {}).get("precision", 0)
    rec  = report.get("1", {}).get("recall",    0)
    f1   = report.get("1", {}).get("f1-score",  0)
    fpr  = 1 - report.get("0", {}).get("recall", 1)

    # ── KPI Strip ──
    st.markdown('<div class="sec-label">Detection Performance</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card" style="--accent-color: var(--cyan)">
        <div class="kpi-label">Detection Accuracy</div>
        <div class="kpi-value">{acc*100:.1f}<span style="font-size:1rem;font-weight:400;color:var(--text-dim)">%</span></div>
        <div class="kpi-sub">Overall classification rate</div>
      </div>
      <div class="kpi-card" style="--accent-color: var(--green)">
        <div class="kpi-label">F1-Score (Attack)</div>
        <div class="kpi-value">{f1:.3f}</div>
        <div class="kpi-sub">Harmonic mean of P&amp;R</div>
      </div>
      <div class="kpi-card" style="--accent-color: var(--amber)">
        <div class="kpi-label">Recall (Sensitivity)</div>
        <div class="kpi-value">{rec*100:.1f}<span style="font-size:1rem;font-weight:400;color:var(--text-dim)">%</span></div>
        <div class="kpi-sub">Threats correctly detected</div>
      </div>
      <div class="kpi-card" style="--accent-color: var(--red)">
        <div class="kpi-label">False Positive Rate</div>
        <div class="kpi-value">{fpr*100:.1f}<span style="font-size:1rem;font-weight:400;color:var(--text-dim)">%</span></div>
        <div class="kpi-sub">Normal traffic flagged</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts Row ──
    left_col, right_col = st.columns([3, 2], gap="medium")

    with left_col:
        st.markdown('<div class="sec-label">Training Convergence</div>', unsafe_allow_html=True)
        with st.container():
            hist_df = pd.DataFrame({
                "Train Accuracy":      history["accuracy"],
                "Validation Accuracy": history["val_accuracy"],
                "Train Loss":          history["loss"],
                "Val Loss":            history["val_loss"],
            })
            tab1, tab2 = st.tabs(["Accuracy", "Loss"])
            with tab1:
                st.line_chart(hist_df[["Train Accuracy", "Validation Accuracy"]],
                              color=["#00c8ff", "#00e8a2"], height=200)
            with tab2:
                st.line_chart(hist_df[["Train Loss", "Val Loss"]],
                              color=["#ff3b5c", "#ffb340"], height=200)

    with right_col:
        st.markdown('<div class="sec-label">Confusion Matrix</div>', unsafe_allow_html=True)
        cm_arr = np.array(cm)
        tn = int(cm_arr[0][0]) if len(cm_arr) > 1 else 0
        fp = int(cm_arr[0][1]) if len(cm_arr) > 1 else 0
        fn = int(cm_arr[1][0]) if len(cm_arr) > 1 else 0
        tp = int(cm_arr[1][1]) if len(cm_arr) > 1 else 0

        st.markdown(f"""
        <div class="cm-grid">
          <div></div>
          <div class="cm-header">Pred Normal</div>
          <div class="cm-header">Pred Attack</div>
          <div class="cm-label">True Normal</div>
          <div class="cm-cell cm-tn">{tn:,}</div>
          <div class="cm-cell cm-fp">{fp:,}</div>
          <div class="cm-label">True Attack</div>
          <div class="cm-cell cm-fn">{fn:,}</div>
          <div class="cm-cell cm-tp">{tp:,}</div>
        </div>
        <div style="margin-top:0.75rem;display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;">
          <div style="font-size:0.6rem;color:var(--green)">✓ True Positive: {tp:,}</div>
          <div style="font-size:0.6rem;color:var(--red)">✗ False Positive: {fp:,}</div>
          <div style="font-size:0.6rem;color:var(--amber)">✗ False Negative: {fn:,}</div>
          <div style="font-size:0.6rem;color:var(--cyan)">✓ True Negative: {tn:,}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="height:0.75rem"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Classification Report</div>', unsafe_allow_html=True)
        report_rows = []
        for cls, label in [("0", "Normal"), ("1", "Attack")]:
            if cls in report:
                r = report[cls]
                report_rows.append({
                    "Class": label,
                    "Precision": f"{r.get('precision',0):.3f}",
                    "Recall":    f"{r.get('recall',0):.3f}",
                    "F1":        f"{r.get('f1-score',0):.3f}",
                    "Support":   f"{int(r.get('support',0)):,}",
                })
        if report_rows:
            st.dataframe(pd.DataFrame(report_rows), hide_index=True, use_container_width=True)

    st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)

    # ── Cross-device generalization check (strict for N-BaIoT) ──
    if data_mode == "nbaiot":
      st.markdown('<div class="sec-label">Cross-Device Generalization Check</div>', unsafe_allow_html=True)
      train_device = st.session_state.get("train_device", "")
      nbaiot_devices_eval = _discover_nbaiot_device_paths(PROJECT_ROOT)
      eval_candidates = sorted([d for d in nbaiot_devices_eval.keys() if d != train_device])

      if train_device:
        st.caption(f"Training device: {train_device}")

      if eval_candidates:
        g1, g2 = st.columns([3, 1])
        with g1:
          eval_device = st.selectbox(
            "Holdout Device (unseen during training)",
            eval_candidates,
            key="eval_device_select"
          )
        with g2:
          st.write("")
          run_gen_eval = st.button("Run Holdout Eval", use_container_width=True)

        if run_gen_eval:
          load_data, train_model, predict_fn, evaluate_model, get_attack_scores, load_saved_artifacts, save_training_meta, use_keras_backend = get_imports()
          eval_paths = nbaiot_devices_eval[eval_device]
          try:
            X_eval, y_eval, _ = load_data(
              mode="nbaiot",
              benign_csv=eval_paths["benign"],
              attack_path=eval_paths["attack"],
            )
            report_eval, cm_eval = evaluate_model(
              X_eval,
              y_eval,
              st.session_state.scaler,
              st.session_state.model,
              seq_len=seq_len,
            )
            st.session_state.gen_eval = {
              "device": eval_device,
              "report": report_eval,
              "cm": cm_eval,
            }
            st.rerun()
          except Exception as e:
            st.error(f"Generalization evaluation failed: {e}")

        if st.session_state.gen_eval:
          g = st.session_state.gen_eval
          gr = g["report"]
          gc = np.array(g["cm"])
          g_acc = float(gr.get("accuracy", 0.0))
          g_f1 = float(gr.get("1", {}).get("f1-score", 0.0))
          g_fpr = 1.0 - float(gr.get("0", {}).get("recall", 1.0))

          c1, c2, c3 = st.columns(3)
          c1.metric("Holdout Accuracy", f"{g_acc*100:.1f}%")
          c2.metric("Holdout F1 (Attack)", f"{g_f1:.3f}")
          c3.metric("Holdout False Positive", f"{g_fpr*100:.1f}%")

          st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
          target_fp = st.slider(
            "Target holdout false positive budget",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            key="target_holdout_fp_budget"
          )

          try:
            eval_paths = nbaiot_devices_eval[g["device"]]
            X_eval_cal, y_eval_cal, _ = load_data(
              mode="nbaiot",
              benign_csv=eval_paths["benign"],
              attack_path=eval_paths["attack"],
            )
            y_true_seq, attack_scores = get_attack_scores(
              X_eval_cal,
              y_eval_cal,
              st.session_state.scaler,
              st.session_state.model,
              seq_len=seq_len,
            )

            neg_scores = attack_scores[y_true_seq == 0]
            if len(neg_scores) > 0:
              calibrated_threshold = float(np.quantile(neg_scores, 1.0 - float(target_fp)))
              y_pred_cal = (attack_scores >= calibrated_threshold).astype(int)

              tn = int(((y_true_seq == 0) & (y_pred_cal == 0)).sum())
              fp = int(((y_true_seq == 0) & (y_pred_cal == 1)).sum())
              fn = int(((y_true_seq == 1) & (y_pred_cal == 0)).sum())
              tp = int(((y_true_seq == 1) & (y_pred_cal == 1)).sum())

              total = max(1, tn + fp + fn + tp)
              cal_acc = (tp + tn) / total
              cal_fpr = fp / max(1, fp + tn)
              cal_precision = tp / max(1, tp + fp)
              cal_recall = tp / max(1, tp + fn)
              cal_f1 = 0.0 if (cal_precision + cal_recall) == 0 else (2 * cal_precision * cal_recall) / (cal_precision + cal_recall)

              k1, k2, k3, k4 = st.columns(4)
              k1.metric("Calibrated Threshold", f"{calibrated_threshold:.3f}")
              k2.metric("Calibrated FPR", f"{cal_fpr*100:.1f}%")
              k3.metric("Calibrated F1", f"{cal_f1:.3f}")
              k4.metric("Calibrated Accuracy", f"{cal_acc*100:.1f}%")

              if st.button("Apply Calibrated Threshold", use_container_width=True):
                profiles = dict(st.session_state.get("threshold_profiles", {}))
                profiles[g["device"]] = float(calibrated_threshold)
                st.session_state["threshold_profiles"] = profiles
                _save_threshold_profiles(profiles)
                st.session_state["iso_threshold_slider"] = float(calibrated_threshold)
                st.session_state["last_profile_device"] = g["device"]
                st.success(
                  f"Isolation threshold updated to {calibrated_threshold:.3f} and saved for {g['device']}"
                )
                st.rerun()
          except Exception as cal_err:
            st.caption(f"Threshold calibration skipped: {cal_err}")

          profiles = st.session_state.get("threshold_profiles", {})
          if profiles:
            st.markdown('<div style="height:0.4rem"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-label">Saved Device Thresholds</div>', unsafe_allow_html=True)
            profiles_df = pd.DataFrame([
              {"Device": d, "Threshold": f"{float(t):.3f}"}
              for d, t in sorted(profiles.items())
            ])
            st.dataframe(profiles_df, hide_index=True, use_container_width=True)

          if g_acc < 0.90 or g_f1 < 0.90:
            st.warning(
              f"Model does not generalize strongly to {g['device']} yet. "
              "This is expected when same-device metrics are optimistic."
            )
          else:
            st.success(f"Strong holdout performance on unseen device: {g['device']}")
      else:
        st.info("No alternate N-BaIoT device available for holdout evaluation.")

      st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)

    # ── Inference ──
    st.markdown('<div class="sec-label">Live Threat Inference &amp; Isolation Engine</div>', unsafe_allow_html=True)

    inf_col1, inf_col2, inf_col3 = st.columns([2, 1, 1])
    with inf_col1:
        n_inf = st.slider("Traffic samples to analyse", 10, 80, 25, key="n_inf_slider")
    with inf_col2:
        st.write("")
        run_inf = st.button("⚡  RUN INFERENCE", use_container_width=True)
    with inf_col3:
        st.write("")
        if st.session_state.inf_results:
            res_df  = pd.DataFrame(st.session_state.inf_results)
            csv_out = res_df.to_csv(index=False)
            st.download_button("⬇  Export CSV", csv_out, "threat_log.csv", "text/csv", use_container_width=True)

    if run_inf:
        load_data, train_model, predict_fn, evaluate_model, get_attack_scores, load_saved_artifacts, save_training_meta, use_keras_backend = get_imports()
        X = st.session_state.X
        with st.spinner("Scanning traffic windows..."):
            start_idx = max(0, len(X) - n_inf - seq_len - 5)
            X_sample  = X[start_idx: start_idx + n_inf + seq_len]
            results   = predict_fn(
                X_sample,
                st.session_state.scaler,
                st.session_state.model,
                seq_len=seq_len,
                isolate_threshold=float(iso_threshold),
            )
        st.session_state.inf_results = results
        st.rerun()

    if st.session_state.inf_results:
        results  = st.session_state.inf_results
        attacks  = [r for r in results if r["prediction"] == 1]
        isolated = [r for r in results if r.get("isolated")]
        normals  = [r for r in results if r["prediction"] == 0]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Samples",      len(results))
        m2.metric("Normal Traffic",     len(normals))
        m3.metric("Threats Detected",   len(attacks))
        m4.metric("Devices Isolated",   len(isolated))

        st.markdown('<div style="height:0.75rem"></div>', unsafe_allow_html=True)

        # Log table header
        st.markdown("""
        <div class="log-header">
          <div class="log-header-cell">Sample</div>
          <div class="log-header-cell">Status</div>
          <div class="log-header-cell">Classification</div>
          <div class="log-header-cell">Confidence</div>
          <div class="log-header-cell">Action</div>
        </div>
        """, unsafe_allow_html=True)

        for r in results:
            conf = r["confidence"]
            is_attack   = r["prediction"] == 1
            is_isolated = r.get("isolated", False)

            if is_isolated:
                row_cls  = "log-entry-isolated"
                badge    = '<span class="badge badge-isolated">⚑ ISOLATED</span>'
                action   = '<span style="color:var(--amber);font-size:0.62rem">QUARANTINE ACTIVE</span>'
                conf_col = "var(--amber)"
            elif is_attack:
                row_cls  = "log-entry-attack"
                badge    = '<span class="badge badge-attack">⚠ ATTACK</span>'
                action   = '<span style="color:var(--red);font-size:0.62rem">LOW CONF — WATCH</span>'
                conf_col = "var(--red)"
            else:
                row_cls  = ""
                badge    = '<span class="badge badge-normal">✓ NORMAL</span>'
                action   = '<span style="color:var(--text-dim);font-size:0.62rem">PASS</span>'
                conf_col = "var(--green)"

            st.markdown(f"""
            <div class="log-entry {row_cls}">
              <div class="log-cell log-idx">#{r['sample_idx']:04d}</div>
              <div class="log-cell">{badge}</div>
              <div class="log-cell" style="color:var(--text-mid);font-size:0.68rem">
                {'Anomalous traffic pattern detected' if is_attack else 'Benign network activity'}
              </div>
              <div class="log-cell">
                <div class="conf-bar-wrap">
                  <div class="conf-bar">
                    <div class="conf-bar-fill" style="width:{conf*100:.0f}%;background:{conf_col};opacity:0.8"></div>
                  </div>
                  <span class="conf-val">{conf*100:.1f}%</span>
                </div>
              </div>
              <div class="log-cell">{action}</div>
            </div>
            """, unsafe_allow_html=True)

        # Summary footer
        n = len(results)
        threat_rate = len(attacks) / n * 100 if n else 0
        st.markdown(f"""
        <div style="margin-top:0.75rem;display:flex;gap:2rem;padding:0.6rem 0.8rem;
                    background:var(--bg-surface);border-radius:3px;border:1px solid var(--border-dim)">
          <span style="font-size:0.62rem;color:var(--text-dim)">
            TOTAL: <span style="color:var(--text-mid)">{n}</span>
          </span>
          <span style="font-size:0.62rem;color:var(--text-dim)">
            THREAT RATE: <span style="color:{'var(--red)' if threat_rate > 20 else 'var(--amber)' if threat_rate > 5 else 'var(--green)'}">{threat_rate:.1f}%</span>
          </span>
          <span style="font-size:0.62rem;color:var(--text-dim)">
            ISO THRESHOLD: <span style="color:var(--cyan)">{iso_threshold:.0%}</span>
          </span>
          <span style="font-size:0.62rem;color:var(--text-dim)">
            MODEL: <span style="color:var(--text-mid)">LSTM · SEQ={seq_len}</span>
          </span>
        </div>
        """, unsafe_allow_html=True)

else:
    # ── Onboarding ──
    st.markdown('<div class="sec-label">Getting Started</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="onboard-grid">
      <div class="onboard-card">
        <div class="onboard-step">01</div>
        <div class="onboard-title">Configure Pipeline</div>
        <div class="onboard-desc">
          Select your data source — use the built-in synthetic dataset for an instant demo,
          or point to your UNSW-NB15 or N-BaIoT CSV files. Tune the LSTM window size,
          epoch count, and isolation confidence threshold above.
        </div>
      </div>
      <div class="onboard-card">
        <div class="onboard-step">02</div>
        <div class="onboard-title">Train the Model</div>
        <div class="onboard-desc">
          Hit <strong style="color:var(--cyan)">▶ TRAIN</strong> to fit the LSTM classifier on
          sliding windows of IoT network traffic. Training metrics — accuracy, F1-score,
          false positive rate, and confusion matrix — are rendered live after completion.
        </div>
      </div>
      <div class="onboard-card">
        <div class="onboard-step">03</div>
        <div class="onboard-title">Detect &amp; Isolate</div>
        <div class="onboard-desc">
          Run inference across traffic windows. The isolation engine automatically
          quarantines devices where the attack confidence exceeds the threshold.
          Export the full threat log as CSV for reporting.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

    arch_toggle_label = "Hide Architecture Overview" if st.session_state.show_architecture else "Show Architecture Overview"
    if st.button(arch_toggle_label, use_container_width=True, key="toggle_architecture_overview"):
        st.session_state.show_architecture = not st.session_state.show_architecture
        st.rerun()

    if st.session_state.show_architecture:
        st.markdown("""
        ```
        IoT Traffic Stream
              │
              ▼
      ┌─────────────────────┐
      │  Data Pre-processing│  Normalise · Denoise · Sequence Windows
      └──────────┬──────────┘
                 │
                 ▼
      ┌─────────────────────┐
      │   LSTM Classifier   │  Temporal anomaly detection (seq_len frames)
      └──────────┬──────────┘
                 │
           ┌─────┴──────┐
           ▼            ▼
        [ NORMAL ]   [ ATTACK ]
           │            │
           │     confidence ≥ threshold?
           │            │
           │      ┌─────┴──────┐
           │      ▼            ▼
           │  [WATCH]    [ISOLATE]  ← device quarantined
           └──────┴────────────┘
                   │
                   ▼
             Threat Log + Export
        ```
        """)
