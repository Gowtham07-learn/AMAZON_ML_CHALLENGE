"""
Blocking Module: 5-Key Inverted Index with Cascading Stopword Pruning & Provenance Ranking
Amazon ML Challenge 2026: Business Entity Resolution
"""

from array import array
from collections import defaultdict
import re
import unicodedata
from typing import Dict, List, Set, Tuple

LEGAL_SUFFIXES = {
    "pvt", "ltd", "private", "limited", "inc", "incorporated",
    "corp", "corporation", "llc", "llp", "gmbh", "sa", "sarl", "co", "company"
}

GENERIC_FIRST_WORDS = {
    "the", "national", "american", "sri", "shree", "om", "star", "prime",
    "global", "general", "royal", "super", "united", "first", "new", "city",
    "apex", "central", "standard", "premier", "universal", "best", "golden"
}

MAX_BLOCK_SIZE = 150
TOP_K_CANDIDATES = 30

def clean_name(name: str) -> str:
    name = str(name or "").strip().lower()
    name = unicodedata.normalize("NFKC", name)
    name = re.sub(r"[^\w\s]", " ", name, flags=re.UNICODE)
    tokens = [t for t in name.split() if t not in LEGAL_SUFFIXES]
    return " ".join(tokens)

def clean_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()

def get_first_token(name: str) -> str:
    tokens = clean_name(name).split()
    for tok in tokens:
        if tok not in GENERIC_FIRST_WORDS and len(tok) >= 3:
            return tok
    return tokens[0] if tokens else ""

def get_region(addr: str) -> str:
    tokens = clean_text(addr).split()
    return tokens[-2] if len(tokens) >= 2 else (tokens[-1] if tokens else "")

def get_postal_prefix(addr: str) -> str:
    tokens = clean_text(addr).split()
    for tok in reversed(tokens):
        if any(c.isdigit() for c in tok) and len(tok) >= 3:
            return tok[:3]
    return ""

class InvertedBlockIndex:
    def __init__(self, max_block_size: int = MAX_BLOCK_SIZE):
        self.max_block_size = max_block_size
        self.block_exact_name = defaultdict(lambda: array("i"))
        self.block_exact_addr = defaultdict(lambda: array("i"))
        self.block_country_first = defaultdict(lambda: array("i"))
        self.block_country_region = defaultdict(lambda: array("i"))
        self.block_postal_pref = defaultdict(lambda: array("i"))
        
    def add_record(self, idx: int, name_clean: str, addr_clean: str, country_clean: str, raw_name: str, raw_addr: str):
        if name_clean:
            arr = self.block_exact_name[name_clean]
            if len(arr) < self.max_block_size:
                arr.append(idx)
                
        if addr_clean:
            arr = self.block_exact_addr[addr_clean]
            if len(arr) < self.max_block_size:
                arr.append(idx)
                
        f_tok = get_first_token(raw_name)
        if country_clean and f_tok:
            arr = self.block_country_first[f"{country_clean}_{f_tok}"]
            if len(arr) < self.max_block_size:
                arr.append(idx)
                
        reg = get_region(raw_addr)
        if country_clean and reg and f_tok:
            arr = self.block_country_region[f"{country_clean}_{reg}_{f_tok}"]
            if len(arr) < self.max_block_size:
                arr.append(idx)
                
        post = get_postal_prefix(raw_addr)
        if post and f_tok:
            arr = self.block_postal_pref[f"{post}_{f_tok}"]
            if len(arr) < self.max_block_size:
                arr.append(idx)

    def query(self, name_clean: str, addr_clean: str, country_clean: str, raw_name: str, raw_addr: str, top_k: int = TOP_K_CANDIDATES) -> List[int]:
        cand_hit_counts = defaultdict(int)
        f_tok = get_first_token(raw_name)
        reg = get_region(raw_addr)
        post = get_postal_prefix(raw_addr)
        
        if name_clean and name_clean in self.block_exact_name:
            for cid in self.block_exact_name[name_clean]:
                cand_hit_counts[cid] += 3
                
        if addr_clean and addr_clean in self.block_exact_addr:
            for cid in self.block_exact_addr[addr_clean]:
                cand_hit_counts[cid] += 3
                
        if country_clean and f_tok:
            k = f"{country_clean}_{f_tok}"
            if k in self.block_country_first:
                for cid in self.block_country_first[k]:
                    cand_hit_counts[cid] += 1
                    
        if country_clean and reg and f_tok:
            k = f"{country_clean}_{reg}_{f_tok}"
            if k in self.block_country_region:
                for cid in self.block_country_region[k]:
                    cand_hit_counts[cid] += 1
                    
        if post and f_tok:
            k = f"{post}_{f_tok}"
            if k in self.block_postal_pref:
                for cid in self.block_postal_pref[k]:
                    cand_hit_counts[cid] += 1
                    
        if not cand_hit_counts:
            return []
            
        sorted_cands = sorted(cand_hit_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [c[0] for c in sorted_cands]
