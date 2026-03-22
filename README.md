# IoT Dynamic Threat Isolation System
**Implementation for: "Dynamic Threat Isolation in IoT-Based Systems"**

---

## Quick Start (< 5 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## Project Structure

```
iot_threat/
├── app.py           ← Streamlit UI (run this)
├── model.py         ← LSTM model definition, training, inference
├── data_loader.py   ← Dataset loaders (demo + real datasets)
├── requirements.txt
└── README.md
```

---

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

---

## Team
1. Rakshith H R — 1BM23IS191  
2. Shreeram Prakash Hegde — 1BM23IS232  
3. Shreevatsa G Uppalli — 1BM23IS234  
4. Sharan Malali — 1BM23IS223
