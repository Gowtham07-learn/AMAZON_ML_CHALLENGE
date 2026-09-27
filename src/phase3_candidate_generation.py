"""
Phase 3 — Candidate Generation / Blocking
Amazon ML Challenge: Multi-source entity resolution.

This script contains the deterministic blocking pipeline used to generate
candidate_pairs.tsv. It is intentionally streaming-oriented because the
candidate set is too large to keep fully in memory.

Sources:
  S1 = train_source1.tsv / test_source1.tsv
  S2 = train_source2.tsv / test_source2.tsv
  S3 = train_source3.tsv / test_source3.tsv

Blocks:
  1. exact_name
  2. exact_address
  3. exact_postal
  4. country_first_name
  5. first_last_name
  6. country_first_last
  7. country_region_first
  8. postal_prefix_first

Oversized block keys are skipped to prevent candidate explosion.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

MAX_BLOCK_SIZE = 5000

BLOCKS = [
    "exact_name",
    "exact_address",
    "exact_postal",
    "country_first_name",
    "first_last_name",
    "country_first_last",
    "country_region_first",
    "postal_prefix_first",
]


def normalize_text(value: str) -> str:
    value = str(value or "").strip().lower()
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def postal_tokens(address: str) -> List[str]:
    return re.findall(r"\b[a-z0-9][a-z0-9 -]{2,9}\b", address.lower())


def get_region(address: str) -> str:
    # Conservative region extraction: use the last comma-separated component
    # before a postal code when available.
    parts = [p.strip() for p in address.split(",") if p.strip()]
    if len(parts) >= 2:
        return parts[-2]
    return ""


def block_key(row: Dict[str, str], block: str) -> str:
    name = normalize_text(row.get("business_name", ""))
    address = normalize_text(row.get("business_address", ""))
    country = normalize_text(row.get("country", ""))

    tokens = name.split()
    first = tokens[0] if tokens else ""
    last = tokens[-1] if tokens else ""

    postal = ""
    for tok in postal_tokens(address):
        if any(ch.isdigit() for ch in tok):
            postal = tok
            break

    if block == "exact_name":
        return name
    if block == "exact_address":
        return address
    if block == "exact_postal":
        return postal
    if block == "country_first_name":
        return f"{country}|{first}"
    if block == "first_last_name":
        return f"{first}|{last}"
    if block == "country_first_last":
        return f"{country}|{first}|{last}"
    if block == "country_region_first":
        region = normalize_text(get_region(address))
        return f"{country}|{region}|{first}"
    if block == "postal_prefix_first":
        prefix = re.sub(r"[^a-z0-9]", "", postal)[:3]
        return f"{prefix}|{first}"

    raise ValueError(f"Unknown block: {block}")


def build_index(
    source_path: Path,
    block: str,
    max_block_size: int = MAX_BLOCK_SIZE,
) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = defaultdict(list)

    with source_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            entity_id = row["entity_id"]
            key = block_key(row, block)
            if not key or key.endswith("|") or key.startswith("|"):
                continue

            bucket = buckets[key]
            if len(bucket) < max_block_size:
                bucket.append(entity_id)

    return dict(buckets)


def generate_candidates(
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    output_path: Path,
    max_block_size: int = MAX_BLOCK_SIZE,
) -> None:
    """
    Generate the final candidate_pairs.tsv.

    Output schema:
      source1_entity_id    candidate_entity_id

    Every emitted pair comes from at least one deterministic block.
    Duplicate pairs are removed per S1 entity.
    """

    print("Building S2/S3 blocking indexes...")

    indexes: Dict[str, Dict[str, List[str]]] = {}
    for block in BLOCKS:
        print(f"  indexing {block}")
        idx2 = build_index(s2_path, block, max_block_size)
        idx3 = build_index(s3_path, block, max_block_size)

        merged: Dict[str, List[str]] = defaultdict(list)
        for key, ids in idx2.items():
            merged[key].extend(ids)
        for key, ids in idx3.items():
            merged[key].extend(ids)

        indexes[block] = dict(merged)
        print(f"    keys={len(indexes[block]):,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_pairs = 0
    total_s1 = 0

    with (
        s1_path.open("r", encoding="utf-8", errors="replace", newline="") as f1,
        output_path.open("w", encoding="utf-8", newline="") as out,
    ):
        reader = csv.DictReader(f1, delimiter="\t")
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for row in reader:
            s1_id = row["entity_id"]
            candidates: Set[str] = set()

            for block in BLOCKS:
                key = block_key(row, block)
                if key in indexes[block]:
                    candidates.update(indexes[block][key])

            if candidates:
                writer.writerow([s1_id, ",".join(sorted(candidates))])
                total_pairs += len(candidates)

            total_s1 += 1
            if total_s1 % 100_000 == 0:
                print(
                    f"S1 processed={total_s1:,} | "
                    f"candidate links={total_pairs:,}"
                )

    print(f"Done. S1 rows={total_s1:,}")
    print(f"Candidate links={total_pairs:,}")
    print(f"Output={output_path}")


if __name__ == "__main__":
    # Change these paths for your local/Colab layout.
    ROOT = Path("/content/AMAZON_ML_CHALLENGE-main")
    DATA = ROOT / "student_resource" / "dataset"

    generate_candidates(
        DATA / "train" / "train_source1.tsv",
        DATA / "train" / "train_source2.tsv",
        DATA / "train" / "train_source3.tsv",
        ROOT / "output" / "candidate_pairs.tsv",
    )
