"""
data_loader.py — Swap this file to use your real dataset.

SUPPORTED DATASETS (just change the load function):
  - UNSW-NB15   → CSV with 'label' or 'attack_cat' column
  - N-BaIoT     → benign CSV  +  attack RAR (contains multiple CSVs)
  - IoTID20     → CSV with 'Label' column
  - KDD Cup 99  → standard benchmark

HOW TO PLUG IN YOUR DATA:
  1. Download your dataset CSV(s)
  2. Replace `load_demo_data()` call in load_data() with the appropriate loader below
  3. Make sure X is a numpy float32 array and y is 0/1 integer labels
"""

import os
import tempfile
import shutil
import subprocess
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────
# DEMO / SYNTHETIC DATA (works out of the box)
# ─────────────────────────────────────────────

def load_demo_data(n_samples=3000, n_features=20, seed=42):
    """
    Generates synthetic IoT traffic data for demo purposes.
    Normal traffic: low variance, low packet size
    Attack traffic: high variance, unusual patterns
    """
    rng = np.random.default_rng(seed)

    # Normal traffic (70%)
    n_normal = int(n_samples * 0.7)
    X_normal = rng.normal(loc=0.3, scale=0.1, size=(n_normal, n_features)).astype(np.float32)

    # Attack traffic (30%) — DoS, port scan, MITM simulated as outliers
    n_attack = n_samples - n_normal
    X_attack = rng.normal(loc=0.8, scale=0.4, size=(n_attack, n_features)).astype(np.float32)
    X_attack += rng.uniform(0, 1, size=X_attack.shape)  # add noise spikes

    X = np.vstack([X_normal, X_attack])
    y = np.array([0] * n_normal + [1] * n_attack, dtype=np.int32)

    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx], [f"feature_{i}" for i in range(n_features)]


# ─────────────────────────────────────────────
# UNSW-NB15 LOADER
# ─────────────────────────────────────────────

def load_unsw_nb15(csv_path: str):
    """
    Load UNSW-NB15 dataset.
    Download: https://research.unsw.edu.au/projects/unsw-nb15-dataset
    """
    unsw_columns = [
        "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes",
        "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts",
        "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
        "Sjit", "Djit", "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",
        "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
        "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
        "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label"
    ]

    df = pd.read_csv(csv_path, low_memory=False)

    # Some UNSW CSV files are headerless. If so, reload with canonical names.
    normalized = {str(c).strip().lower(): c for c in df.columns}
    if "label" not in normalized and df.shape[1] == len(unsw_columns):
        df = pd.read_csv(csv_path, low_memory=False, header=None, names=unsw_columns)
        normalized = {str(c).strip().lower(): c for c in df.columns}

    # Resolve label column robustly (label/Label or last binary column fallback).
    label_col = normalized.get("label")
    if label_col is None:
        candidate = df.columns[-1]
        cand_vals = pd.to_numeric(df[candidate], errors="coerce").dropna()
        unique_vals = set(cand_vals.unique().tolist())
        if unique_vals and unique_vals.issubset({0, 1}):
            label_col = candidate
        else:
            raise ValueError(
                "UNSW loader could not find a label column. "
                f"Detected columns sample: {list(df.columns[:8])}"
            )

    y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int).values

    # Drop non-feature columns
    drop_norm = {"label", "attack_cat", "srcip", "dstip", "sport", "dsport"}
    drop_cols = [col for col in df.columns if str(col).strip().lower() in drop_norm]
    X_df = df.drop(columns=drop_cols)

    # Encode categoricals
    for col in X_df.select_dtypes(include='object').columns:
        X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))

    X = X_df.values.astype(np.float32)
    feature_names = X_df.columns.tolist()
    return X, y, feature_names


# ─────────────────────────────────────────────
# N-BaIoT LOADER
# ─────────────────────────────────────────────

def _read_attack_csvs_from_rar(rar_path: str) -> pd.DataFrame:
    """
    Extract and concatenate all CSV files found inside a RAR archive.
    Requires: pip install rarfile
    Also needs the `unrar` system binary:
      - Windows: install WinRAR or UnRAR and ensure it's on PATH
      - Linux:   sudo apt install unrar
      - macOS:   brew install rar
    """
    try:
        import rarfile
    except ImportError:
        raise ImportError(
            "rarfile not installed. Run:  pip install rarfile\n"
            "Also install the unrar binary for your OS."
        )

    frames = []
    failed_members = []

    with rarfile.RarFile(rar_path) as rf:
        csv_names = [f for f in rf.namelist() if f.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV files found inside {rar_path}")

        print(f"  Found {len(csv_names)} CSV(s) in RAR: {csv_names}")
        for name in csv_names:
            # First attempt: stream directly from the archive.
            try:
                with rf.open(name) as fh:
                    df = pd.read_csv(fh)
                    df["_source_file"] = os.path.basename(name)
                    frames.append(df)
                    print(f"    Loaded {name}: {len(df):,} rows")
                continue
            except Exception as stream_err:
                print(f"    Stream read failed for {name}: {stream_err}")

            # Second attempt: extract this member to temp file then read.
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    rf.extract(name, path=tmpdir)
                    extracted_path = os.path.join(tmpdir, name)
                    df = pd.read_csv(extracted_path)
                    df["_source_file"] = os.path.basename(name)
                    frames.append(df)
                    print(f"    Loaded via extract fallback {name}: {len(df):,} rows")
            except Exception as extract_err:
                failed_members.append((name, str(extract_err)))
                print(f"    Extract fallback failed for {name}: {extract_err}")

    if not frames:
        details = "; ".join([f"{n}: {e}" for n, e in failed_members[:3]])
        raise ValueError(
            "Could not read any CSV files from the attack RAR archive. "
            "The archive may be incomplete/corrupted or unsupported by the installed unrar backend. "
            "Please extract the RAR manually and provide the extracted folder path in the app. "
            f"Sample errors: {details}"
        )

    if failed_members:
        print(f"  Warning: skipped {len(failed_members)} unreadable CSV(s) from RAR")

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Total attack rows after combining: {len(combined):,}")
    return combined


def _read_attack_csvs_from_folder(folder_path: str) -> pd.DataFrame:
    """
    Fallback: if you already extracted the RAR, point to the folder
    containing the CSVs instead. Reads all *.csv files in that folder.
    """
    csv_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".csv")
    ]
    if not csv_files:
        raise ValueError(f"No CSV files found in folder: {folder_path}")

    frames = []
    for path in sorted(csv_files):
        df = pd.read_csv(path)
        df["_source_file"] = os.path.basename(path)
        frames.append(df)
        print(f"  Loaded {os.path.basename(path)}: {len(df):,} rows")

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Total attack rows: {len(combined):,}")
    return combined


def _read_attack_csvs_from_rar_bsdtar(rar_path: str) -> pd.DataFrame:
    """
    Extract RAR with bsdtar as a robust fallback on Windows setups where
    rarfile/unrar streaming may fail with short-read errors.
    """
    bsdtar_bin = shutil.which("bsdtar")
    if not bsdtar_bin:
        raise RuntimeError("bsdtar not found on PATH")

    with tempfile.TemporaryDirectory() as tmpdir:
        proc = subprocess.run(
            [bsdtar_bin, "-xf", rar_path, "-C", tmpdir],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            raise RuntimeError(f"bsdtar extraction failed: {stderr or stdout or 'unknown error'}")

        csv_files = []
        for root, _, files in os.walk(tmpdir):
            for f in files:
                if f.lower().endswith(".csv"):
                    csv_files.append(os.path.join(root, f))

        if not csv_files:
            raise ValueError(f"No CSV files found after extracting {rar_path} with bsdtar")

        frames = []
        for path in sorted(csv_files):
            df = pd.read_csv(path)
            df["_source_file"] = os.path.basename(path)
            frames.append(df)
            print(f"    Loaded via bsdtar {os.path.basename(path)}: {len(df):,} rows")

        combined = pd.concat(frames, ignore_index=True)
        print(f"  Total attack rows after bsdtar fallback: {len(combined):,}")
        return combined


def load_nbaiot(benign_csv: str, attack_path: str):
    """
    Load N-BaIoT dataset.
    Download: https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot

    Args:
        benign_csv:   path to the benign traffic CSV file
        attack_path:  EITHER:
                        - path to the attack .rar file  (e.g. gafgyt_attacks.rar)
                        - path to a folder containing the extracted attack CSVs
                      The function auto-detects which one you passed.

    The attack RAR typically contains CSVs like:
        combo.csv, junk.csv, scan.csv, tcp.csv, udp.csv
    All of them are concatenated and labelled as attack (label=1).
    """
    # ── Benign data ──────────────────────────────────────────
    print("Loading benign traffic...")
    df_benign = pd.read_csv(benign_csv)
    print(f"  Benign rows: {len(df_benign):,}")

    # ── Attack data — RAR or extracted folder ─────────────────
    print(f"Loading attack traffic from: {attack_path}")
    if os.path.isdir(attack_path):
        df_attack = _read_attack_csvs_from_folder(attack_path)
    elif attack_path.lower().endswith(".rar"):
        try:
            df_attack = _read_attack_csvs_from_rar(attack_path)
        except Exception as rar_err:
            print(f"  RAR streaming path failed: {rar_err}")
            print("  Trying bsdtar extraction fallback...")
            df_attack = _read_attack_csvs_from_rar_bsdtar(attack_path)
    else:
        # Single CSV fallback (original behaviour)
        df_attack = pd.read_csv(attack_path)
        print(f"  Attack rows: {len(df_attack):,}")

    # ── Drop internal bookkeeping column if present ───────────
    for df in [df_benign, df_attack]:
        if "_source_file" in df.columns:
            df.drop(columns=["_source_file"], inplace=True)

    # ── Align columns — both files must share the same features ─
    benign_cols = set(df_benign.columns)
    attack_cols = set(df_attack.columns)
    shared_cols = sorted(benign_cols & attack_cols)

    if not shared_cols:
        raise ValueError(
            "Benign and attack CSVs share no common columns. "
            "Make sure they come from the same device in the N-BaIoT dataset."
        )

    dropped = (benign_cols | attack_cols) - set(shared_cols)
    if dropped:
        print(f"  Dropped non-shared columns: {dropped}")

    df_benign = df_benign[shared_cols].copy()
    df_attack = df_attack[shared_cols].copy()

    # ── Label and combine ────────────────────────────────────
    df_benign["label"] = 0
    df_attack["label"] = 1

    df = pd.concat([df_benign, df_attack], ignore_index=True).sample(frac=1, random_state=42)
    df.reset_index(drop=True, inplace=True)

    y = df["label"].astype(int).values
    X_df = df.drop(columns=["label"])

    # Handle any remaining non-numeric columns
    for col in X_df.select_dtypes(include="object").columns:
        X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))

    X = X_df.values.astype(np.float32)
    feature_names = X_df.columns.tolist()

    print(f"Final dataset: {X.shape[0]:,} samples × {X.shape[1]} features  "
          f"| Normal: {(y==0).sum():,}  Attack: {(y==1).sum():,}")
    return X, y, feature_names


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def load_data(mode="demo", **kwargs):
    """
    mode options:
      "demo"     → synthetic data, no files needed
      "unsw"     → pass csv_path=...
      "nbaiot"   → pass benign_csv=..., attack_path=...
                   attack_path can be a .rar file OR an extracted folder
    """
    if mode == "demo":
        return load_demo_data(**kwargs)
    elif mode == "unsw":
        return load_unsw_nb15(kwargs["csv_path"])
    elif mode == "nbaiot":
        return load_nbaiot(kwargs["benign_csv"], kwargs["attack_path"])
    else:
        raise ValueError(f"Unknown mode: {mode}")