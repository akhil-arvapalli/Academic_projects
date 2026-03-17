from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
from rapidfuzz import fuzz

from .config import DedupConfig


def _add_pair(pairs: List[Dict], left_id: int, right_id: int, reason: str, score: int) -> None:
    if left_id == right_id:
        return
    a, b = sorted([int(left_id), int(right_id)])
    pairs.append(
        {
            "left_student_id": a,
            "right_student_id": b,
            "reason": reason,
            "score": score,
        }
    )


def _pairs_from_group(group: pd.DataFrame, reason: str, score: int = 100) -> List[Dict]:
    ids = group["student_id"].tolist()
    out: List[Dict] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            _add_pair(out, ids[i], ids[j], reason, score)
    return out


def detect_duplicates(df: pd.DataFrame, config: DedupConfig) -> pd.DataFrame:
    pairs: List[Dict] = []

    for _, grp in df.groupby("email_norm"):
        if grp["email_norm"].iloc[0] and len(grp) > 1:
            pairs.extend(_pairs_from_group(grp, "exact_email"))

    for _, grp in df.groupby("roll_no_norm"):
        if grp["roll_no_norm"].iloc[0] and len(grp) > 1:
            pairs.extend(_pairs_from_group(grp, "exact_roll_no"))

    for _, grp in df.groupby("phone_norm"):
        if grp["phone_norm"].iloc[0] and len(grp) > 1:
            pairs.extend(_pairs_from_group(grp, "exact_phone"))

    for _, grp in df.groupby(["full_name_norm", "dob_norm"]):
        if grp["full_name_norm"].iloc[0] and len(grp) > 1:
            pairs.extend(_pairs_from_group(grp, "name_plus_dob"))

    records = df.to_dict("records")
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            left = records[i]
            right = records[j]
            same_dob = left["dob_norm"] and left["dob_norm"] == right["dob_norm"]
            same_phone = left["phone_norm"] and left["phone_norm"] == right["phone_norm"]
            if not (same_dob or same_phone):
                continue
            score = fuzz.token_sort_ratio(left["full_name_norm"], right["full_name_norm"])
            if score >= config.name_similarity_threshold:
                _add_pair(pairs, left["student_id"], right["student_id"], "fuzzy_name", int(score))

    pair_df = pd.DataFrame(pairs)
    if pair_df.empty:
        return pair_df

    pair_df = pair_df.drop_duplicates(subset=["left_student_id", "right_student_id", "reason"])
    return pair_df.sort_values(["left_student_id", "right_student_id", "reason"]).reset_index(drop=True)


def choose_canonical_records(df: pd.DataFrame, pair_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if pair_df.empty:
        keep = df.copy()
        removed = df.iloc[0:0].copy()
        return keep, removed

    parent: Dict[int, int] = {int(v): int(v) for v in df["student_id"].tolist()}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for _, row in pair_df.iterrows():
        union(int(row["left_student_id"]), int(row["right_student_id"]))

    groups: Dict[int, List[int]] = {}
    for sid in df["student_id"].tolist():
        root = find(int(sid))
        groups.setdefault(root, []).append(int(sid))

    to_drop = set()
    for ids in groups.values():
        if len(ids) > 1:
            ids_sorted = sorted(ids)
            for sid in ids_sorted[1:]:
                to_drop.add(sid)

    keep_df = df[~df["student_id"].isin(to_drop)].copy()
    removed_df = df[df["student_id"].isin(to_drop)].copy()
    return keep_df, removed_df
