# Phase 4: Lean Blocking Recall Evaluation Report

## 1. Executive Summary
- **Candidate File**: `candidate_pairs.tsv`
- **Candidate File Size**: **0.76 GB** (819,618,422 bytes)
- **Global Pair-Level Blocking Recall**: **62.58%**
- **Global Entity-Level Coverage**: **88.23%**
- **True Matches Captured**: **4,779,994** / 7,638,365
- **Total S1 Queries Evaluated**: 2,206,821
- **Total Candidate Links Generated**: 61,212,492
- **Average Candidates / S1**: **27.75**

## 2. Recall Breakdown by Entity
- **Full Recall (100% true matches found)**: 764,617 (34.65%)
- **Partial Recall (>=1 true match found)**: 1,182,512 (53.58%)
- **Zero Recall (0 true matches found)**: 259,692 (11.77%)

## 3. Blocking Configuration (`TIGHT_BLOCKS`)
- `exact_name`
- `exact_address`
- `country_first_name`
- `country_region_first`
- `postal_prefix_first`

- **Bucket Cap**: `MAX_BLOCK_SIZE = 150`
- **Per-Entity Cap**: `TOP_K_CANDIDATES = 30` (Ranked by Provenance Count)

## 4. Sample Missed Matches
- `Source 1 (S1-773889195)` <--> `Matched (S2-970528089)`
- `Source 1 (S1-773889195)` <--> `Matched (S3-202893869)`
- `Source 1 (S1-377745466)` <--> `Matched (S3-402918963)`
- `Source 1 (S1-133037285)` <--> `Matched (S3-183080822)`
- `Source 1 (S1-755362802)` <--> `Matched (S3-440254853)`
- `Source 1 (S1-851869949)` <--> `Matched (S3-949828938)`
- `Source 1 (S1-629417405)` <--> `Matched (S3-786131946)`
- `Source 1 (S1-629417405)` <--> `Matched (S2-928426462)`
- `Source 1 (S1-629417405)` <--> `Matched (S3-438606634)`
- `Source 1 (S1-629417405)` <--> `Matched (S3-606982601)`
- `Source 1 (S1-22305073)` <--> `Matched (S2-578011604)`
- `Source 1 (S1-504790211)` <--> `Matched (S2-482219248)`
- `Source 1 (S1-504790211)` <--> `Matched (S3-956253301)`
- `Source 1 (S1-564729135)` <--> `Matched (S3-209102076)`
- `Source 1 (S1-564729135)` <--> `Matched (S2-256254095)`
