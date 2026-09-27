"""
Feature Extraction Module: 12 High-Signal Domain Features
Amazon ML Challenge 2026: Business Entity Resolution
"""

import re
from typing import List

def levenshtein_sim(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    len1, len2 = len(s1), len(s2)
    max_len = max(len1, len2)
    if abs(len1 - len2) / max_len > 0.7:
        return 0.2
    if len1 > len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1
    distances = list(range(len1 + 1))
    for i2, c2 in enumerate(s2):
        new_dist = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                new_dist.append(distances[i1])
            else:
                new_dist.append(1 + min((distances[i1], distances[i1 + 1], new_dist[-1])))
        distances = new_dist
    return max(0.0, 1.0 - (distances[-1] / max_len))

def extract_pairwise_features(s1_name_clean: str, s1_addr_clean: str, s1_country: str,
                              c_name_clean: str, c_addr_clean: str, c_country: str, c_eid: str) -> List[float]:
    t1_name = set(s1_name_clean.split())
    t2_name = set(c_name_clean.split())
    t1_addr = set(s1_addr_clean.split())
    t2_addr = set(c_addr_clean.split())
    
    # F1: Name Jaccard
    name_jaccard = len(t1_name & t2_name) / max(1, len(t1_name | t2_name))
    # F2: Name Levenshtein
    name_lev = levenshtein_sim(s1_name_clean, c_name_clean)
    # F3: Name Token Sort Ratio
    n1_sort = " ".join(sorted(t1_name))
    n2_sort = " ".join(sorted(t2_name))
    name_token_sort = levenshtein_sim(n1_sort, n2_sort)
    # F4: Exact Name Match
    name_exact = 1.0 if (s1_name_clean and s1_name_clean == c_name_clean) else 0.0
    # F5: Name Prefix (4 chars)
    name_prefix = 1.0 if (s1_name_clean[:4] and s1_name_clean[:4] == c_name_clean[:4]) else 0.0
    # F6: Address Jaccard
    addr_jaccard = len(t1_addr & t2_addr) / max(1, len(t1_addr | t2_addr))
    # F7: Address Digits Overlap (Building/Plot numbers)
    d1 = set(re.findall(r"\b\d+\b", s1_addr_clean))
    d2 = set(re.findall(r"\b\d+\b", c_addr_clean))
    digits_overlap = len(d1 & d2) / max(1, len(d1 | d2)) if (d1 or d2) else 0.5
    # F8: Postal Exact Match
    p1 = set(re.findall(r"\b[a-z0-9]{3,8}\b", s1_addr_clean))
    p2 = set(re.findall(r"\b[a-z0-9]{3,8}\b", c_addr_clean))
    postal_exact = 1.0 if (p1 and p2 and (p1 & p2)) else 0.0
    # F9: Postal Prefix (first 3 digits)
    pref1 = {tok[:3] for tok in p1 if any(c.isdigit() for c in tok)}
    pref2 = {tok[:3] for tok in p2 if any(c.isdigit() for c in tok)}
    postal_pref = 1.0 if (pref1 and pref2 and (pref1 & pref2)) else 0.0
    # F10: Country Match
    country_match = 1.0 if (s1_country and c_country and s1_country == c_country) else (0.0 if (s1_country and c_country) else 0.5)
    # F11: Address Length Ratio
    addr_len_ratio = min(len(s1_addr_clean), len(c_addr_clean)) / max(1, max(len(s1_addr_clean), len(c_addr_clean)))
    # F12: Source Pair Type
    source_type = 1.0 if c_eid.startswith("S2") else 0.0
    
    return [
        name_jaccard, name_lev, name_token_sort, name_exact, name_prefix,
        addr_jaccard, digits_overlap, postal_exact, postal_pref,
        country_match, addr_len_ratio, source_type
    ]
