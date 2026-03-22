# IoT Dynamic Threat Isolation System

Implementation for the project Dynamic Threat Isolation in IoT-Based Systems.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

## Project Files

```text
.
|- app.py                # Streamlit interface and workflow
|- model.py              # Training, evaluation, inference logic
|- data_loader.py        # Dataset loading and preprocessing utilities
|- requirements.txt      # Python dependencies
|- README.md
|- .gitignore
|- data/                 # local-only datasets (ignored by Git)
|- saved_model.keras     # local model artifact (ignored by Git)
|- scaler.pkl            # local scaler artifact (ignored by Git)
|- training_meta.json    # local training metadata (ignored by Git)
```

## Datasets

Supported sources:
1. Demo synthetic dataset (quick verification)
2. UNSW-NB15 CSV
3. N-BaIoT device traffic CSVs

Dataset references:
1. UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. N-BaIoT: https://archive.ics.uci.edu/dataset/442

## What the App Does

1. Builds sequence windows from IoT traffic features
2. Trains an LSTM classifier for normal vs attack detection
3. Supports holdout evaluation for cross-device generalization checks
4. Applies threshold-based isolation for attack responses
5. Reports metrics including confusion matrix and false positive behavior

## Important Git Policy

This repository intentionally tracks code and documentation only.

Ignored local-only assets:
1. Full dataset folders under data/
2. Raw archives such as .rar and .zip
3. Trained model files such as .keras and .h5
4. Preprocessing/model artifacts such as .pkl and training metadata JSON

If you train locally, artifacts stay on your machine and are not pushed.

## Using Your Real Dataset

### UNSW-NB15
1. Download from: https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. In the sidebar → Dataset → select "UNSW-NB15 (CSV)"
3. Enter the path to the CSV file

### N-BaIoT
1. Download from: https://archive.ics.uci.edu/dataset/442
2. In the sidebar → Dataset → select "N-BaIoT (CSV)"
3. Enter paths to benign and attack CSV files

### Demo mode (default)
- No files needed — synthetic IoT traffic is generated automatically
- Good enough for a live demo

---

## What It Implements (from the paper)

| Paper Component                  | Implementation                              |
|----------------------------------|---------------------------------------------|
| Temporal feature extraction      | LSTM on sliding windows of traffic features |
| Threat detection                 | Binary classifier (normal vs attack)        |
| Automated isolation              | Confidence-threshold based flagging         |
| Multi-dataset evaluation         | Plug-in loaders for UNSW-NB15, N-BaIoT     |
| Detection metrics                | Accuracy, Precision, Recall, F1, CM        |

> **Simplified for demo:** Federated learning, graph convolution (DyGCN),
> and SDN-based isolation rules are described in the paper but not
> implemented here — they require a real IoT network infrastructure.

---

## Adjustable Parameters (sidebar)

| Parameter           | Default | Description                              |
|---------------------|---------|------------------------------------------|
| Sequence Length     | 10      | Number of traffic samples per LSTM input |
| Max Epochs          | 15      | Training iterations                      |
| Isolation Threshold | 0.75    | Minimum confidence to isolate a device   |
