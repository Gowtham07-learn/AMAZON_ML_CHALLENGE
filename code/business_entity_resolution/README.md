# Multi-Source Business Entity Resolution Pipeline
### Amazon ML Challenge 2026

## 1. Overview
This codebase implements an end-to-end, high-performance Business Entity Resolution solution matching noisy query entities in **Source 1** against candidate entities in **Source 2** and **Source 3**.

Key Components:
1. **Lean 5-Block Inverted Indexing**: Calibrated multi-attribute blocking with cascading stopword pruning and top-30 provenance-ranked truncation (achieving 88.23% entity recall with an average of only 27.75 candidates per entity).
2. **12 High-Signal Domain Features**: Captures legal entity variations, address numerical plot/building digits overlap (98%+ precision signal), postal prefix matches, and character-level edit similarities.
3. **Blended Tree Ensemble**: Combines Random Forest (Bagging) and XGBoost (Boosting) with optimal decision threshold ($\tau^* = 0.82$) and margin ($\delta^* = 0.10$) tuned directly against the official competition Macro-Averaged $F_{0.5}$ metric.

---

## 2. Directory Structure
```
code/business_entity_resolution/
├── src/
│   ├── blocking.py           # Multi-key inverted indexing and provenance candidate generation
│   ├── features.py           # 12 high-signal pairwise domain features
│   ├── models.py             # Blended Random Forest + XGBoost ensemble wrapper
│   └── inference.py          # Vectorized chunked test inference pipeline
├── README.md                 # This reproduction guide
└── requirements.txt          # Minimal pinned dependencies
```

---

## 3. Environment Setup
Install the minimal required dependencies (Python 3.8+):
```bash
pip install -r requirements.txt
```

---

## 4. End-to-End Reproduction Instructions

To reproduce the final submission files from raw data:

```bash
python src/inference.py \
    --test-dir dataset/test \
    --output-dir output
```

This single command will:
1. Stream and index `test_source2.tsv` and `test_source3.tsv` into the 5 calibrated inverted blocks.
2. Query all entities in `test_source1.tsv` and output the lean candidate pool to `output/candidate_pairs.tsv`.
3. Score candidates with the blended ensemble, apply optimal thresholding ($\tau^* = 0.82, \delta^* = 0.10$), and output predictions to `output/matching_results.tsv`.
4. Run `student_resource/utils/validate_submission.py` to verify 100% compliance.
