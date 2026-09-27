# Amazon ML Challenge 2026: Multi-Source Business Entity Resolution

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Macro F0.5](https://img.shields.io/badge/Macro_F0.5-0.7269-brightgreen.svg)]()
[![Candidate Set](https://img.shields.io/badge/Candidate_Pool-0.76_GB-green.svg)]()
[![Official Validator](https://img.shields.io/badge/Validator-PASS_(Exit_0)-success.svg)]()

This repository contains the complete, production-grade winning solution for the **Amazon ML Challenge 2026 (Business Entity Resolution Track)**. The challenge requires resolving multi-source business entities across **Source 1** (query anchor) against large-scale records in **Source 2** and **Source 3** across multiple countries (United States, India, and France).

---

## 🏆 Key Achievements & Benchmarks

- **Lean Candidate Generation (Maximum Reduction Ratio)**:
  - Compressed the candidate pool from an unconstrained naive **45+ GB** down to just **0.76 GB** (819 MB / 61.2M candidate links).
  - Maintained an average of only **27.75 candidates per Source 1 entity** (strictly within competition constraints).
  - Preserved a high **88.23% entity-level coverage ceiling** across all 2,206,821 Source 1 training queries.
- **Blended Ensemble with Direct Macro $F_{0.5}$ Tuning**:
  - Blended **Random Forest** (Bagging) and **XGBoost** (Boosting, `tree_method='hist'`) across 12 high-signal domain features (legal suffix stripping, building/plot number overlap, postal prefix, Levenshtein, and Jaccard).
  - Swept thresholds directly against the official per-entity **Macro-Averaged $F_{0.5}$ metric**:
    - **Optimal Decision Threshold**: $\tau^* = 0.82$
    - **Optimal Margin Threshold**: $\delta^* = 0.10$
    - **Singleton Accuracy**: Boosted to **0.7756** (+7.7 points) by eliminating false merges on singletons.
    - **Validated Macro $F_{0.5}$**: **0.7269**.
- **High-Throughput Streaming Engine**:
  - Full-scale streaming throughput of **~6,500 – 9,000 queries per second**.
  - All 1,732,544 Test Source 1 entities processed in **under 4.5 minutes**.
  - Peak memory strictly bounded under **1.2 GB RAM** using 32-bit signed C-integer mappings (`array('i')`).
- **Official Compliance**:
  - Tested and 100% compliant with `student_resource/utils/validate_submission.py` (**`PASS — Exit Code 0`**).

---

## 📂 Repository Structure

```
AMAZON_ML_CHALLENGE/
├── code/
│   └── business_entity_resolution/        # Standalone runnable reproduction pipeline
│       ├── src/
│       │   ├── blocking.py                # 5-block inverted index with cascading stopword pruning
│       │   ├── features.py                # 12 high-signal pairwise domain features
│       │   ├── models.py                  # Blended Random Forest + XGBoost ensemble wrapper
│       │   └── inference.py               # Vectorized chunked test inference pipeline
│       ├── README.md                      # Pipeline reproduction guide
│       └── requirements.txt               # Minimal pinned dependencies
├── src/                                   # Root source code package
│   ├── blocking.py
│   ├── features.py
│   ├── models.py
│   └── inference.py
├── notebooks/                             # Interactive Jupyter development notebooks
│   ├── 01_data_audit.ipynb                # Phase 1: Data exploration & noise profiling
│   ├── 02_feature_extraction.ipynb        # Phase 2: Feature engineering library
│   ├── 03_feature_exploration.ipynb       # Phase 2: Similarity correlation analysis
│   ├── 04_candidate_analysis.ipynb        # Phase 3: Inverted blocking & index calibration
│   ├── 05_recall_analysis.ipynb           # Phase 4: Candidate pool recall evaluation
│   ├── 06_model_training.ipynb            # Phase 5 & 6: Ensemble training & threshold grid search
│   └── 07_submission_inference.ipynb      # Phase 7: Test inference & output validation
├── reports/                               # Comprehensive analysis reports
│   ├── phase1/audit_report.md             # Data hygiene & schema audit findings
│   └── phase4/recall_report.md            # Blocking recall & candidate coverage benchmarks
├── student_resource/
│   ├── Documentation_template.md          # Official filled methodology report
│   ├── README.md                          # Hackathon instructions & format rules
│   └── utils/
│       └── validate_submission.py         # Official competition format validator
├── output/                                # Test output deliverables (git-ignored)
│   ├── candidate_pairs.tsv                # Test blocking candidate set (604 MB)
│   └── matching_results.tsv               # Final leaderboard predictions (249 MB)
├── requirements.txt                       # Python dependencies
└── README.md                              # This documentation
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
Clone the repository and install dependencies in Python 3.8+:
```bash
git clone https://github.com/Gowtham07-learn/AMAZON_ML_CHALLENGE.git
cd AMAZON_ML_CHALLENGE
pip install -r requirements.txt
```

### 2. End-to-End Test Set Inference
To reproduce the final submission files from raw data:
```bash
python src/inference.py \
    --test-dir student_resource/dataset/test \
    --output-dir output \
    --models-dir data/processed/models
```

### 3. Submission Format Validation
Verify that generated outputs are 100% compliant with competition scoring rules:
```bash
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```
*Expected output: `PASS — no blocking issues found. Safe to submit.`*

---

## 🔬 Methodology Summary

### 1. Inverted Blocking Index (`blocking.py`)
To avoid $O(N \times M)$ combinatorial explosion, records are indexed across **5 calibrated tight blocks**:
1. `exact_name`: Normalized NFKC, lowercased, legal entity suffix-stripped string.
2. `exact_address`: Normalized, punctuation-cleaned full address.
3. `country_first_name`: Country code + first salient name token (skipping generic stopwords `the, national, american, sri, shree, om, star, prime, global, royal, super`).
4. `country_region_first`: Country + extracted administrative region + first salient name token.
5. `postal_prefix_first`: 3-digit postal prefix + first salient name token.

**Pruning Rules**: Bucket cap `MAX_BLOCK_SIZE = 150`, per-entity cap `TOP_K_CANDIDATES = 30` (ranked by multi-block provenance hit frequency).

### 2. Feature Engineering (`features.py`)
12 domain features designed to maximize precision:
- **Name**: Jaccard similarity, Levenshtein distance, token sort ratio, exact match indicator, 4-char prefix match.
- **Address**: Token Jaccard similarity, address length ratio, exact postal match, 3-digit postal prefix match.
- **High-Precision Signal**: Numerical plot/building number overlap (`digits_overlap`) carries >98% precision signal.
- **Context**: Country consistency indicator and source indicator ($S_2$ vs $S_3$).

### 3. Model Architecture & $F_{0.5}$ Optimization (`models.py`)
- **Random Forest**: 100 estimators, `max_depth=16`, `min_samples_split=10`, `n_jobs=-1`.
- **XGBoost**: 150 estimators, `max_depth=7`, `learning_rate=0.08`, `tree_method='hist'`.
- **Blend**: $P_{\text{blend}} = 0.50 \cdot P_{\text{XGBoost}} + 0.50 \cdot P_{\text{RandomForest}}$.
- **Thresholds**: $\tau^* = 0.82$, $\delta^* = 0.10$. Unconfident entities are output with empty matches, capturing a full **1.0 point on all singletons**.

---

## 👥 Authors
- **Gowtham**
- **Dharaneesh**

*Amazon ML Challenge 2026*
