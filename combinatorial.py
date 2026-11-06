"""Exact combinatorial features and finite Category B hypothesis search.

Only NumPy and the Python standard library are needed.  The null distribution
is computed from every sorted six-element subset of 1..45, never simulated.
The finite grammar is fixed by this module; no holdout value sets a threshold.
"""
from __future__ import annotations

from itertools import combinations, islice
from pathlib import Path
import csv
import json
import math
import time

import numpy as np


TOTAL = math.comb(45, 6)
MODULI = (2, 3, 4, 5, 7, 9, 10, 11)
PRIMES = frozenset((2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43))
SQUARES = frozenset((1, 4, 9, 16, 25, 36))
FIBONACCI = frozenset((1, 2, 3, 5, 8, 13, 21, 34))
PAIR_INDEX = tuple(combinations(range(6), 2))
JOINT_PAIRS = (
    ("ac", "max_gap"), ("ac", "span"), ("min_gap", "span"),
    ("sum", "odd_count"), ("sum", "low_count"),
    ("prime_count", "gcd_pairs"), ("consecutive_count", "same_tail_pairs"),
    ("tail_sum", "same_tail_pairs"), ("prime_count", "odd_count"),
    ("max_gap", "consecutive_count"), ("low_count", "odd_count"),
    ("ac", "same_tail_pairs"),
)
_LOOKUPS = {
    "prime_count": np.array([i in PRIMES for i in range(46)], dtype=np.int8),
    "composite_count": np.array([i > 1 and i not in PRIMES for i in range(46)], dtype=np.int8),
    "square_count": np.array([i in SQUARES for i in range(46)], dtype=np.int8),
    "fib_count": np.array([i in FIBONACCI for i in range(46)], dtype=np.int8),
    "multiple3_count": np.array([i > 0 and i % 3 == 0 for i in range(46)], dtype=np.int8),
    "multiple5_count": np.array([i > 0 and i % 5 == 0 for i in range(46)], dtype=np.int8),
}
_GCD = np.gcd(np.arange(46)[:, None], np.arange(46)[None, :]) > 1


def static_features(x: np.ndarray) -> dict[str, np.ndarray]:
    """Features for sorted, distinct six-number rows; variance uses ddof=0."""
    x = np.asarray(x, dtype=np.int16)
    if x.ndim != 2 or x.shape[1] != 6:
        raise ValueError("x must have shape (n, 6)")
    n = len(x)
    gaps = np.diff(x, axis=1)
    tails = x % 10
    diffs = np.stack([x[:, j] - x[:, i] for i, j in PAIR_INDEX], axis=1)
    sorted_diffs = np.sort(diffs, axis=1)
    out = {f"x{i+1}": x[:, i] for i in range(6)}
    out.update({"sum": x.sum(axis=1), "odd_count": (x % 2).sum(axis=1),
                "low_count": (x <= 22).sum(axis=1)})
    out.update({name: lookup[x].sum(axis=1) for name, lookup in _LOOKUPS.items()})
    out["tail_sum"] = tails.sum(axis=1)
    out["same_tail_pairs"] = sum((tails[:, i] == tails[:, j]).astype(np.int8)
                                  for i, j in PAIR_INDEX)
    for d in range(5):
        out[f"decade{d}_count"] = (x // 10 == d).sum(axis=1)
    out["gcd_pairs"] = sum(_GCD[x[:, i], x[:, j]].astype(np.int8) for i, j in PAIR_INDEX)
    out.update({f"gap{i+1}": gaps[:, i] for i in range(5)})
    out.update({"min_gap": gaps.min(axis=1), "max_gap": gaps.max(axis=1),
                "span": x[:, -1] - x[:, 0],
                "consecutive_count": (gaps == 1).sum(axis=1),
                "ac": 1 + (sorted_diffs[:, 1:] != sorted_diffs[:, :-1]).sum(axis=1) - 5,
                "mean": x.mean(axis=1), "variance": x.var(axis=1)})
    assert all(len(v) == n for v in out.values())
    return out


def evaluate_spec(features: dict[str, np.ndarray], spec: dict) -> np.ndarray:
    """Evaluate the JSON-compatible finite rule grammar on feature arrays."""
    values = []
    for term in spec["terms"]:
        x = features[term["feature"]]
        op, value = term["op"], term["value"]
        if op == "le":
            y = x <= value
        elif op == "ge":
            y = x >= value
        elif op == "eq":
            y = x == value
        elif op == "between":
            y = (x >= value[0]) & (x <= value[1])
        elif op == "modin":
            y = np.isin(x % term["mod"], value)
        else:
            raise ValueError(f"Unknown operator: {op}")
        values.append(y)
    if not values:
        raise ValueError("A rule needs at least one term")
    return np.logical_or.reduce(values) if spec.get("logic", "and") == "or" else np.logical_and.reduce(values)


def _integer_names() -> list[str]:
    return [k for k in static_features(np.array([[1, 2, 3, 4, 5, 6]])) if k not in ("mean", "variance")]


def combination_chunks(chunk_size: int = 100_000):
    """Enumerate each sorted combination exactly once with bounded memory."""
    source = combinations(range(1, 46), 6)
    while True:
        batch = list(islice(source, chunk_size))
        if not batch:
            break
        yield np.asarray(batch, dtype=np.int16)


def enumerate_null(cache_dir: Path) -> dict:
    """Return exact integer marginal and joint histograms of all 8,145,060 rows."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "exact_null_v1.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            total = int(data["total"])
            if total != TOTAL:
                raise ValueError("Unexpected null-cache size")
            hist = {k[3:]: data[k] for k in data.files if k.startswith("h__")}
            joint = {k[3:]: data[k] for k in data.files if k.startswith("j__")}
            assert all(int(v.sum()) == TOTAL for v in (*hist.values(), *joint.values()))
            return {"hist": hist, "joint": joint, "total": TOTAL, "cache": str(path)}
    start = time.perf_counter()
    hist = {k: np.zeros(280, dtype=np.int64) for k in _integer_names()}
    joint = {f"{a}|{b}": np.zeros((280, 280), dtype=np.int64) for a, b in JOINT_PAIRS}
    counted = 0
    for batch in combination_chunks():
        features = static_features(batch)
        for key, target in hist.items():
            target += np.bincount(features[key].astype(np.int64), minlength=280)
        for a, b in JOINT_PAIRS:
            index = features[a].astype(np.int64) * 280 + features[b]
            joint[f"{a}|{b}"] += np.bincount(index, minlength=280 * 280).reshape(280, 280)
        counted += len(batch)
        if counted % 1_000_000 == 0:
            print(f"Exact null: {counted:,}/{TOTAL:,} ({time.perf_counter()-start:.1f}s)", flush=True)
    assert counted == TOTAL
    # Crop only trailing zero bins; all array indices remain the actual values.
    hist = {k: v[:np.flatnonzero(v)[-1]+1] for k, v in hist.items()}
    joint = {k: v[:np.flatnonzero(v.sum(axis=1))[-1]+1, :np.flatnonzero(v.sum(axis=0))[-1]+1]
             for k, v in joint.items()}
    assert all(int(v.sum()) == TOTAL for v in (*hist.values(), *joint.values()))
    payload = {"total": np.array(TOTAL), **{f"h__{k}": v for k, v in hist.items()},
               **{f"j__{k}": v for k, v in joint.items()}}
    np.savez_compressed(path, **payload)
    return {"hist": hist, "joint": joint, "total": TOTAL, "cache": str(path),
            "enumeration_seconds": time.perf_counter() - start}


def _term(feature, op, value, mod=None):
    result = {"feature": feature, "op": op, "value": value}
    if mod is not None:
        result["mod"] = mod
    return result


def _label(term):
    key, op, value = term["feature"], term["op"], term["value"]
    if op == "le":
        return f"{key} ≤ {value}"
    if op == "ge":
        return f"{key} ≥ {value}"
    if op == "eq":
        return f"{key} = {value}"
    if op == "between":
        return f"{value[0]} ≤ {key} ≤ {value[1]}"
    return f"{key} mod {term['mod']} ∈ {{{','.join(map(str, value))}}}"


def _residue_sets(mod):
    """Fixed single residues, cyclic half-blocks, and their complements."""
    seen = set()
    for width in sorted({1, mod // 2, (mod + 1) // 2, mod - 1}):
        for start in range(mod):
            values = tuple(sorted((start + j) % mod for j in range(width)))
            if values not in seen:
                seen.add(values)
                yield list(values)


def _single_terms(key, hist, include_intervals):
    support = np.flatnonzero(hist)
    for value in support:
        yield _term(key, "le", int(value))
        yield _term(key, "ge", int(value))
    if include_intervals:
        # Exclude edge intervals represented by le/ge; one-value intervals allowed.
        for i, lo in enumerate(support[1:-1], start=1):
            for hi in support[i:-1]:
                yield _term(key, "between", [int(lo), int(hi)])
    for mod in MODULI:
        for residues in _residue_sets(mod):
            yield _term(key, "modin", residues, mod)


def _term_hist_count(hist, term):
    # Used only for event registry creation (small) and tests, not search hot loop.
    indices = np.arange(len(hist))
    return int(hist[evaluate_spec({term["feature"]: indices}, {"terms": [term]})].sum())


def baseline_events(history_x, null):
    """Fixed unary threshold/mod targets with exact baseline in [0.40, 0.50]."""
    features = static_features(history_x)
    result = []
    seen = set()
    for key, hist in null["hist"].items():
        for term in _single_terms(key, hist, include_intervals=False):
            probability = _term_hist_count(hist, term) / TOTAL
            if 0.4 <= probability <= 0.5:
                spec = {"terms": [term], "logic": "and"}
                mask = evaluate_spec(features, spec)
                # Deduplicate identical formulas after canonical JSON encoding;
                # do not deduplicate by historical outcome vectors.
                canonical = json.dumps(spec, sort_keys=True)
                if canonical not in seen:
                    seen.add(canonical)
                    result.append({"name": _label(term), "spec": spec,
                                   "baseline": probability, "mask": mask})
    return result


def search_b(history_x: np.ndarray, train_mask: np.ndarray, cache_dir: Path, coverage_limit=0.5):
    """Search finite mathematical filters using empirical lift and validation rate.

    null['searched_count'] counts all eligible theoretical-coverage hypotheses
    (strictly positive and at most ``coverage_limit``). Definitions are fixed by
    the grammar, while retention explicitly uses validation performance.
    """
    null = enumerate_null(cache_dir)
    features = static_features(history_x)
    train_mask = np.asarray(train_mask, dtype=bool)
    valid_mask = ~train_mask
    nt, nv = int(train_mask.sum()), int(valid_mask.sum())
    rows, searched = [], 0
    nearest = []

    def add(spec, null_k, train_k, valid_k):
        nonlocal searched
        if not (0 < null_k < TOTAL and null_k / TOTAL <= coverage_limit):
            return
        searched += 1
        logic = spec.get("logic", "and")
        name = (" AND " if logic == "and" else " OR ").join(_label(t) for t in spec["terms"])
        record = {"category": "B", "name": name, "spec": spec,
                  "baseline": null_k / TOTAL, "null_count": int(null_k),
                  "train_n": nt, "train_k": int(train_k),
                  "valid_n": nv, "valid_k": int(valid_k),
                  "total_n": nt + nv, "total_k": int(train_k + valid_k)}
        record["train_rate"] = float(train_k / nt)
        record["valid_rate"] = float(valid_k / nv)
        record["total_rate"] = float((train_k + valid_k) / (nt + nv))
        record["lift"] = record["total_rate"] / record["baseline"]
        # Reporting examples are ranked solely on training hits, with a formula
        # tie-breaker. Validation rates are displayed but never used to pick them.
        if len(nearest) < 5 or train_k >= nearest[-1]["train_k"]:
            nearest.append(record)
            nearest.sort(key=lambda r: (-r["train_k"], r["name"]))
            del nearest[5:]
        if nt >= 100 and nv >= 30 and 100 * valid_k >= 55 * nv and record["lift"] > 1.05:
            rows.append(record)

    for key, hist in null["hist"].items():
        train_hist = np.bincount(features[key][train_mask].astype(int), minlength=len(hist))
        valid_hist = np.bincount(features[key][valid_mask].astype(int), minlength=len(hist))
        stacked = np.stack((hist, train_hist, valid_hist))
        cumulative = np.pad(np.cumsum(stacked, axis=1), ((0, 0), (1, 0)))
        indices = np.arange(len(hist))
        for term in _single_terms(key, hist, include_intervals=True):
            value, op = term["value"], term["op"]
            if op == "le":
                counts = cumulative[:, value+1]
            elif op == "ge":
                counts = cumulative[:, -1] - cumulative[:, value]
            elif op == "between":
                counts = cumulative[:, value[1]+1] - cumulative[:, value[0]]
            else:
                selected = np.isin(indices % term["mod"], value)
                counts = stacked[:, selected].sum(axis=1)
            add({"terms": [term], "logic": "and"}, *map(int, counts))

    for a, b in JOINT_PAIRS:
        table = null["joint"][f"{a}|{b}"]
        shape = table.shape
        def empirical(mask):
            code = features[a][mask] * shape[1] + features[b][mask]
            return np.bincount(code.astype(int), minlength=shape[0]*shape[1]).reshape(shape)
        stacked = np.stack((table, empirical(train_mask), empirical(valid_mask)))
        cum = np.pad(stacked.cumsum(axis=1).cumsum(axis=2), ((0, 0), (1, 0), (1, 0)))
        totals = stacked.sum(axis=(1, 2))
        support_a = np.flatnonzero(table.sum(axis=1))
        support_b = np.flatnonzero(table.sum(axis=0))
        for ka in support_a:
            for kb in support_b:
                # Each threshold divides at a value and value+1, covering all
                # nonempty quadrants and their complementary unions.
                both_le = cum[:, ka+1, kb+1]
                a_le = cum[:, ka+1, -1]
                b_le = cum[:, -1, kb+1]
                quadrants = (
                    ("le", int(ka), "le", int(kb), both_le),
                    ("le", int(ka), "ge", int(kb+1), a_le - both_le),
                    ("ge", int(ka+1), "le", int(kb), b_le - both_le),
                    ("ge", int(ka+1), "ge", int(kb+1), totals - a_le - b_le + both_le),
                )
                for opa, va, opb, vb, count in quadrants:
                    terms = [_term(a, opa, va), _term(b, opb, vb)]
                    add({"terms": terms, "logic": "and"}, *map(int, count))
                    # Complement of a conjunction is a union of complements.
                    ca = _term(a, "ge" if opa == "le" else "le", va+1 if opa == "le" else va-1)
                    cb = _term(b, "ge" if opb == "le" else "le", vb+1 if opb == "le" else vb-1)
                    add({"terms": [ca, cb], "logic": "or"}, *map(int, totals-count))
    null["searched_count"] = searched
    null["eligible_count"] = len(rows)
    null["coverage_limit"] = coverage_limit
    null["nearest_rules"] = nearest
    path = Path(cache_dir).parent / "b_nearest_rules.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["category", "name", "baseline", "train_k", "train_n", "train_rate",
              "valid_k", "valid_n", "valid_rate", "total_k", "total_n", "total_rate",
              "null_count", "lift", "spec", "status"]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in nearest:
            exported = dict(row)
            for split in ("train", "valid", "total"):
                exported[f"{split}_rate"] = row[f"{split}_k"] / row[f"{split}_n"]
            exported["spec"] = json.dumps(row["spec"], ensure_ascii=False)
            exported["status"] = "학습 적중률 순 근접 사례; 채택 규칙 아님"
            writer.writerow(exported)
    null["nearest_rules_csv"] = str(path)
    return rows, null


def self_test(null=None):
    """Independent scalar checks on arithmetic and generic evaluator boundaries."""
    examples = np.array([[1, 2, 3, 4, 5, 6], [2, 11, 21, 31, 41, 45],
                         [5, 10, 15, 20, 25, 30], [16, 18, 27, 36, 38, 42]])
    f = static_features(examples)
    for i, row in enumerate(examples.tolist()):
        pairs = list(combinations(row, 2))
        expected = {
            "sum": sum(row), "odd_count": sum(v % 2 for v in row),
            "low_count": sum(v <= 22 for v in row),
            "prime_count": sum(v > 1 and all(v % d for d in range(2, math.isqrt(v)+1)) for v in row),
            "composite_count": sum(v > 1 and any(v % d == 0 for d in range(2, math.isqrt(v)+1)) for v in row),
            "square_count": sum(math.isqrt(v)**2 == v for v in row),
            "same_tail_pairs": sum(a % 10 == b % 10 for a, b in pairs),
            "gcd_pairs": sum(math.gcd(a, b) > 1 for a, b in pairs),
            "ac": len({b-a for a, b in pairs}) - 5,
            "consecutive_count": sum(b-a == 1 for a, b in zip(row, row[1:])),
        }
        for key, expected_value in expected.items():
            assert f[key][i] == expected_value, (row, key, f[key][i], expected_value)
    for term in (_term("sum", "le", 100), _term("sum", "between", [100, 200]),
                 _term("sum", "modin", [0, 1], 3)):
        actual = evaluate_spec(f, {"terms": [term]})
        assert actual.dtype == bool and actual.shape == (4,)
    assert TOTAL == 8_145_060
    if null is not None:
        hist = null["hist"]
        assert all(int(v.sum()) == TOTAL for v in (*hist.values(), *null["joint"].values()))
        assert sum(k * int(v) for k, v in enumerate(hist["sum"])) == 138 * TOTAL
        assert np.array_equal(hist["sum"][21:256], hist["sum"][21:256][::-1])
        assert all(int(hist["odd_count"][k]) == math.comb(23, k) * math.comb(22, 6-k) for k in range(7))
        assert all(int(hist["prime_count"][k]) == math.comb(14, k) * math.comb(31, 6-k) for k in range(7))
        assert int(hist["consecutive_count"][0]) == math.comb(40, 6)
    return {"examples": len(examples), "integer_features": len(_integer_names()),
            "joint_pairs": len(JOINT_PAIRS), "exact_null_checked": null is not None}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    print(self_test())
    if not args.self_test:
        result = enumerate_null(args.cache_dir)
        print(self_test(result))
        print({k: v for k, v in result.items() if k not in ("hist", "joint")})
