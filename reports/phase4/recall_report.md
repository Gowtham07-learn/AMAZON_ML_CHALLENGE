# Phase 4: Blocking Recall Evaluation Report

## 1. Executive Summary
- **Candidate File**: `candidate_pairs.tsv`
- **Pair-Level Recall**: **8.61%**
- **Entity-Level Coverage**: **31.41%**
- **True Matches Captured**: 29,411 / 341,710
- **Evaluated S1 Entities**: 98,649
- **Average Candidates / Entity**: 571.38

## 2. Blocking Configuration (FINAL_BLOCKS)
The candidate generation used the following 6 deterministic blocks:
- `exact_name`
- `exact_address`
- `exact_postal`
- `country_first_name`
- `country_region_first`
- `postal_prefix_first`

Oversized block cap: `MAX_BLOCK_SIZE = 5000`

## 3. Sample Missed Matches
- `Source 1 (S1-925783039)` <--> `Matched (S2-517291332)`
- `Source 1 (S1-925783039)` <--> `Matched (S3-698172821)`
- `Source 1 (S1-925783039)` <--> `Matched (S3-997698194)`
- `Source 1 (S1-925783039)` <--> `Matched (S2-157377754)`
- `Source 1 (S1-773889195)` <--> `Matched (S3-622232873)`
- `Source 1 (S1-773889195)` <--> `Matched (S3-202893869)`
- `Source 1 (S1-773889195)` <--> `Matched (S3-161452820)`
- `Source 1 (S1-773889195)` <--> `Matched (S3-189140669)`
- `Source 1 (S1-773889195)` <--> `Matched (S2-374107005)`
- `Source 1 (S1-773889195)` <--> `Matched (S2-970528089)`
