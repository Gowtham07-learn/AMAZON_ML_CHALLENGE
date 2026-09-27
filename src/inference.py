"""
Inference Module: End-to-End Test Set Inference & Leaderboard Submission Generation
Amazon ML Challenge 2026: Business Entity Resolution
"""

import argparse
from collections import defaultdict
import gc
from pathlib import Path
import sys
import time

import numpy as np

from blocking import InvertedBlockIndex, clean_name, clean_text, TOP_K_CANDIDATES
from features import extract_pairwise_features
from models import BlendedEnsemble

CHUNK_PAIR_LIMIT = 100000

def run_inference(test_dir: Path, output_dir: Path, models_dir: Path):
    print("=" * 70)
    print("Starting Multi-Source Entity Resolution Test Inference")
    print("=" * 70)
    t0_start = time.time()
    
    test_s1 = test_dir / "test_source1.tsv"
    test_s2 = test_dir / "test_source2.tsv"
    test_s3 = test_dir / "test_source3.tsv"
    
    cand_out = output_dir / "candidate_pairs.tsv"
    match_out = output_dir / "matching_results.tsv"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Ensemble
    print("Loading Blended Ensemble and Thresholds...")
    ensemble = BlendedEnsemble.load(models_dir)
    print(f"Loaded Config: w_xgb={ensemble.w_xgb:.2f}, w_rf={ensemble.w_rf:.2f}, tau*={ensemble.threshold}, margin={ensemble.margin_threshold}")
    
    # 2. Build Inverted Index
    print("Indexing Test Source 2 & 3...")
    t0 = time.time()
    index = InvertedBlockIndex()
    cand_eids = []
    cand_names = []
    cand_addrs = []
    cand_countries = []
    
    idx = 0
    for file_path in [test_s2, test_s3]:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                eid, raw_name, raw_addr, raw_country = parts[0], parts[1], parts[2], parts[3]
                n_clean = clean_name(raw_name)
                a_clean = clean_text(raw_addr)
                c_clean = clean_text(raw_country)
                
                cand_eids.append(eid)
                cand_names.append(n_clean)
                cand_addrs.append(a_clean)
                cand_countries.append(c_clean)
                index.add_record(idx, n_clean, a_clean, c_clean, raw_name, raw_addr)
                idx += 1
                
    print(f"Indexed {idx:,} candidate records in {time.time()-t0:.1f}s")
    gc.collect()

    # 3. Stream Test Source 1 & Vectorized Chunked Inference
    print("Streaming Test S1 queries & generating outputs...")
    out_cand_f = open(cand_out, "w", encoding="utf-8", newline="")
    out_match_f = open(match_out, "w", encoding="utf-8", newline="")
    
    out_cand_f.write("source1_entity_id\tcandidate_entity_ids\n")
    out_match_f.write("source1_entity_id\tmatched_entity_ids\n")
    
    chunk_s1_ids = []
    chunk_features = []
    chunk_pair_info = []
    total_queries = 0
    total_matches = 0
    
    def flush():
        nonlocal total_matches
        if not chunk_s1_ids:
            return
        passing_by_s1 = defaultdict(list)
        if chunk_features:
            X = np.array(chunk_features, dtype=np.float32)
            probs = ensemble.predict_proba(X)
            for (s1_idx, c_eid), prob in zip(chunk_pair_info, probs):
                if prob >= ensemble.threshold:
                    passing_by_s1[s1_idx].append((c_eid, float(prob)))
                    
        for s1_idx, s1_id in enumerate(chunk_s1_ids):
            matches = passing_by_s1.get(s1_idx, [])
            if not matches:
                out_match_f.write(f"{s1_id}\t\n")
            else:
                matches.sort(key=lambda x: x[1], reverse=True)
                top_p = matches[0][1]
                kept = [m[0] for m in matches if (top_p - m[1]) <= ensemble.margin_threshold]
                out_match_f.write(f"{s1_id}\t{','.join(kept)}\n")
                total_matches += len(kept)
                
        chunk_s1_ids.clear()
        chunk_features.clear()
        chunk_pair_info.clear()

    with open(test_s1, "r", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            s1_id, raw_name, raw_addr, raw_country = parts[0], parts[1], parts[2], parts[3]
            total_queries += 1
            n_clean = clean_name(raw_name)
            a_clean = clean_text(raw_addr)
            c_clean = clean_text(raw_country)
            
            top_cand_indices = index.query(n_clean, a_clean, c_clean, raw_name, raw_addr)
            cur_idx = len(chunk_s1_ids)
            chunk_s1_ids.append(s1_id)
            
            if not top_cand_indices:
                out_cand_f.write(f"{s1_id}\t\n")
            else:
                top_cand_eids = [cand_eids[i] for i in top_cand_indices]
                out_cand_f.write(f"{s1_id}\t{','.join(top_cand_eids)}\n")
                for c_idx, c_eid in zip(top_cand_indices, top_cand_eids):
                    feats = extract_pairwise_features(
                        n_clean, a_clean, c_clean,
                        cand_names[c_idx], cand_addrs[c_idx], cand_countries[c_idx], c_eid
                    )
                    chunk_features.append(feats)
                    chunk_pair_info.append((cur_idx, c_eid))
                    
            if len(chunk_features) >= CHUNK_PAIR_LIMIT:
                flush()
                
    flush()
    out_cand_f.close()
    out_match_f.close()
    print(f"Completed {total_queries:,} queries in {time.time()-t0_start:.1f}s! Outputs written to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("dataset/test"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--models-dir", type=Path, default=Path("data/processed/models"))
    args = parser.parse_args()
    run_inference(args.test_dir, args.output_dir, args.models_dir)
